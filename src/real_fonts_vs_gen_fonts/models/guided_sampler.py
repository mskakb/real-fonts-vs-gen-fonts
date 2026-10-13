"""Classifier-guided DDIM sampler using a pretrained RealnessScorer.

Guidance formula (applied to x̂_0):
    x̂_0_guided = x̂_0 + realness_scale * (1 - ᾱ_t) / √ᾱ_t * ∇_{x_t} log s(x̂_0)

Equivalent to modifying the noise prediction:
    ε̃ = ε_θ - realness_scale * √(1 - ᾱ_t) * ∇_{x_t} log s(x̂_0)

Reference: Dhariwal & Nichol (2021), "Diffusion Models Beat GANs on Image Synthesis"
"""

from __future__ import annotations

import torch
from torch import Tensor

from real_fonts_vs_gen_fonts.models.diffusion import GaussianDiffusion, unnormalize_to_zero_to_one
from real_fonts_vs_gen_fonts.models.scorer import RealnessScorer


class ClassifierGuidedSampler:
    """DDIM sampler with Classifier Guidance from RealnessScorer.

    Implements a custom sampling loop (without @torch.no_grad) so that
    ∇_{x_t} log s(x̂_0(x_t)) can be computed via autograd.

    Args:
        diffusion: Trained GaussianDiffusion (eval mode, on target device).
        scorer: Trained RealnessScorer (eval mode, on target device).
        realness_scale: Guidance strength.
            >0 guides toward real fonts, <0 toward generated (boundary exploration).
        sampling_timesteps: Number of DDIM denoising steps.
        clip_denoised: Whether to clip x̂_0 to [-1, 1] at each step.
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        scorer: RealnessScorer,
        realness_scale: float = 1.0,
        sampling_timesteps: int = 50,
        clip_denoised: bool = True,
        use_sdf: bool = False,
    ) -> None:
        self.diffusion = diffusion
        self.scorer = scorer
        self.realness_scale = realness_scale
        self.sampling_timesteps = sampling_timesteps
        self.clip_denoised = clip_denoised
        self.use_sdf = use_sdf

    def _time_schedule(self) -> list[tuple[int, int]]:
        total = self.diffusion.num_timesteps
        times = torch.linspace(-1, total - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        return list(zip(times[:-1], times[1:]))

    def _ddim_step(
        self,
        x_start: Tensor,
        pred_noise: Tensor,
        time: int,
        time_next: int,
    ) -> Tensor:
        """Standard DDIM update (no gradient needed)."""
        alpha = self.diffusion.alphas_cumprod[time]
        alpha_next = self.diffusion.alphas_cumprod[time_next]
        sigma = self.diffusion.ddim_sampling_eta * (
            (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
        ).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        noise = torch.randn_like(x_start)
        return x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

    def _apply_guidance(
        self,
        x_t: Tensor,
        x_start: Tensor,
        pred_noise: Tensor,
        time: int,
    ) -> tuple[Tensor, Tensor]:
        """Compute gradient of log s(x̂_0) w.r.t. x_t and update x̂_0."""
        alpha = self.diffusion.alphas_cumprod[time]

        # Scorer expects [0, 1] float32; x̂_0 is in [-1, 1]
        x_start_01 = unnormalize_to_zero_to_one(x_start)
        scores = self.scorer(x_start_01.float())           # (B,)
        log_s = scores.clamp(1e-6, 1 - 1e-6).log().sum()
        grad = torch.autograd.grad(log_s, x_t)[0].detach()

        # Modify x̂_0 and derive consistent ε̃
        x_start_guided = x_start.detach() + self.realness_scale * (1 - alpha) / alpha.sqrt() * grad
        if self.clip_denoised:
            x_start_guided = x_start_guided.clamp(-1, 1)
        pred_noise_guided = (x_t.detach() - alpha.sqrt() * x_start_guided) / (1 - alpha).sqrt()

        return x_start_guided, pred_noise_guided

    def sample(
        self,
        classes: Tensor,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
    ) -> Tensor:
        """Generate images with classifier guidance.

        Args:
            classes: (B,) class indices on the target device.
            cond_scale: CFG++ guidance scale.
            rescaled_phi: CFG++ rescaling factor.

        Returns:
            (B, C, H, W) images in [0, 1].
        """
        diffusion = self.diffusion
        device = diffusion.betas.device
        B = classes.shape[0]
        shape = (B, diffusion.channels, diffusion.image_size, diffusion.image_size)

        img = torch.randn(shape, device=device)

        for time, time_next in self._time_schedule():
            time_cond = torch.full((B,), time, device=device, dtype=torch.long)

            # Temporarily attach grad to x_t for guidance computation
            img_in = img.detach().requires_grad_(self.realness_scale != 0)

            with torch.enable_grad():
                pred_noise, x_start, *_ = diffusion.model_predictions(
                    img_in, time_cond, classes,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                    clip_x_start=self.clip_denoised,
                )

                if self.realness_scale != 0:
                    x_start, pred_noise = self._apply_guidance(
                        img_in, x_start, pred_noise, time
                    )

            x_start = x_start.detach()
            pred_noise = pred_noise.detach()

            if time_next < 0:
                img = x_start
                continue

            with torch.no_grad():
                img = self._ddim_step(x_start, pred_noise, time, time_next)

        return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def score_landscape(
        self,
        classes: Tensor,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
    ) -> tuple[list[int], Tensor]:
        """Run DDIM sampling and record s(x̂_0) at every timestep.

        Useful for understanding how the scorer responds across noise levels
        and whether guidance has room to operate.

        Args:
            classes: (B,) class indices.
            cond_scale: CFG++ guidance scale (no gradient computed here).
            rescaled_phi: CFG++ rescaling factor.

        Returns:
            timesteps: t values in decreasing order, length n_steps.
            scores: (n_steps, B) scorer outputs at each step.
        """
        diffusion = self.diffusion
        device = diffusion.betas.device
        B = classes.shape[0]
        shape = (B, diffusion.channels, diffusion.image_size, diffusion.image_size)

        img = torch.randn(shape, device=device)
        timesteps: list[int] = []
        all_scores: list[Tensor] = []

        for time, time_next in self._time_schedule():
            time_cond = torch.full((B,), time, device=device, dtype=torch.long)

            pred_noise, x_start, *_ = diffusion.model_predictions(
                img, time_cond, classes,
                cond_scale=cond_scale,
                rescaled_phi=rescaled_phi,
                clip_x_start=True,
            )

            x_start_01 = unnormalize_to_zero_to_one(x_start)
            if self.use_sdf:
                from real_fonts_vs_gen_fonts.sdf import sdf_to_binary
                x_start_01 = sdf_to_binary(x_start_01)
            scores = self.scorer(x_start_01.float())   # (B,)
            timesteps.append(time)
            all_scores.append(scores.cpu())

            if time_next < 0:
                break

            img = self._ddim_step(x_start, pred_noise, time, time_next)

        return timesteps, torch.stack(all_scores, dim=0)  # (n_steps, B)
