"""Gradient-free inpainting of Diffusion Policy action chunks (phase-2 smoothing between plans).

Why inpainting: when a new chunk replaces the running one, the two plans rarely agree (diffusion policies
are multimodal), so the robot can jump between strategies at every switch. Real-time chunking (RTC; Black,
Galliker, Levine 2025; for flow-matching policies in ``lerobot/policies/rtc``) treats the next chunk as an
inpainting problem: the actions that are executed while the chunk is computed are *frozen* to the running
plan, and the rest of the overlap is pulled towards it with decaying weights.

Why gradient-free: RTC enforces the constraint with ΠGDM guidance, i.e. a vector-Jacobian product (one
backward pass through the denoiser) per denoising step. On the CPU Brain that multiplies the 0.34 s
inference and breaks the 0.7 s latency budget. For the DDIM-sampled Diffusion Policy we use the classic
replacement method instead (Diffuser, Janner et al. 2022; RePaint, Lugmayr et al. 2022), applied to the
model's prediction of the clean chunk: at every denoising step

    x0' = w * guide + (1 - w) * x0_pred

and the scheduler continues from ``x0'`` (the noise prediction handed to it is recomputed so that the
scheduler's own update reproduces ``x0'``). Weight-1 entries end exactly at the guide, weight-0 entries are
bitwise identical to LeRobot's sampler, and every following step denoises the guided sample, so the free
part of the chunk adapts to the frozen prefix. Cost: a few tensor operations per step, no extra network
passes.

Everything here uses LeRobot's own modules (UNet, noise scheduler, observation encoder) without modifying
them; ``guided_conditional_sample`` mirrors ``DiffusionModel.conditional_sample`` of LeRobot 0.6.1.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class GuidanceError(ValueError):
    """The guide cannot be applied (wrong shapes/weights or unsupported scheduler)."""


def blend_model_output(
    scheduler: Any, model_output: Tensor, sample: Tensor, timestep: Tensor, known: Tensor, weights: Tensor
) -> Tensor:
    """Model output whose implied clean sample is ``w * known + (1 - w) * x0_pred`` where ``w > 0``.

    ``sample``/``model_output``/``known``: ``(B, horizon, action_dim)`` in normalized action space;
    ``weights``: ``(B, horizon, 1)`` in [0, 1]. Entries with weight 0 are returned unchanged.
    """
    prediction_type = scheduler.config.prediction_type
    alpha_prod = scheduler.alphas_cumprod[int(timestep)].to(device=sample.device, dtype=sample.dtype)
    beta_prod = 1.0 - alpha_prod
    if prediction_type == "epsilon":
        x0 = (sample - beta_prod.sqrt() * model_output) / alpha_prod.sqrt()
    elif prediction_type == "sample":
        x0 = model_output
    else:
        raise GuidanceError(f"inpainting supports epsilon or sample prediction, not {prediction_type!r}")
    if getattr(scheduler.config, "clip_sample", False):
        limit = float(scheduler.config.clip_sample_range)
        x0 = x0.clamp(-limit, limit)
    x0_guided = weights * known + (1.0 - weights) * x0
    if prediction_type == "epsilon":
        guided = (sample - alpha_prod.sqrt() * x0_guided) / beta_prod.sqrt()
    else:
        guided = x0_guided
    return torch.where(weights > 0, guided, model_output)


def guided_conditional_sample(
    model: Any, global_cond: Tensor, noise: Tensor, known: Tensor, weights: Tensor
) -> Tensor:
    """LeRobot's ``DiffusionModel.conditional_sample`` with inpainting of the guided entries.

    ``model`` is a LeRobot ``DiffusionModel`` (``policy.diffusion``); ``noise`` the initial sample
    ``(B, horizon, action_dim)``. With all weights 0 the result equals ``model.conditional_sample``.
    """
    if noise.shape != known.shape:
        raise GuidanceError(f"guide shape {tuple(known.shape)} does not match the chunk {tuple(noise.shape)}")
    if weights.shape != (*noise.shape[:2], 1):
        raise GuidanceError(f"weights must have shape {(*noise.shape[:2], 1)}, got {tuple(weights.shape)}")
    scheduler = model.noise_scheduler
    if getattr(scheduler.config, "thresholding", False):
        raise GuidanceError("dynamic thresholding is not supported by the inpainting sampler")
    scheduler.set_timesteps(model.num_inference_steps)
    guided_entries = bool((weights > 0).any())
    sample = noise
    for t in scheduler.timesteps:
        model_output = model.unet(
            sample,
            torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
            global_cond=global_cond,
        )
        if guided_entries:
            model_output = blend_model_output(scheduler, model_output, sample, t, known, weights)
        sample = scheduler.step(model_output, t, sample, generator=None).prev_sample
    return sample
