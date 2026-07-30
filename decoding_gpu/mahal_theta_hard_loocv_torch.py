"""GPU hard relative-bin leave-one-out Mahalanobis decoding."""

from __future__ import annotations

import numpy as np
import torch

from mahal_theta_torch import DecodeConfig, _covdiag_batched, _mahalanobis_batched


@torch.inference_mode()
def mahal_hard_loocv(
    features: np.ndarray,
    theta: np.ndarray,
    config: DecodeConfig,
    bin_half_width: float,
    n_templates: int = 12,
    device: str = "auto",
    dtype: str = "float32",
    return_trialwise: bool = False,
):
    """Decode continuous angles with MATLAB-style relative hard bins and LOOCV.

    For every held-out trial, relative template centres are defined around
    that trial's true angle. Training trials inside each hard circular window
    are averaged, and both covariance and templates exclude the held-out trial.
    """
    if n_templates < 4:
        raise ValueError("n_templates must be at least 4")
    if not 0 < bin_half_width <= np.pi:
        raise ValueError("bin_half_width must be in (0, pi]")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch cannot access a CUDA device")

    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    target = torch.device(device)
    data_t = torch.as_tensor(features, dtype=torch_dtype, device=target)
    theta_t = torch.remainder(
        torch.as_tensor(theta, dtype=torch_dtype, device=target).flatten(),
        2 * torch.pi,
    )
    if data_t.ndim != 3:
        raise ValueError("features must have shape trial x feature x time")
    if theta_t.numel() != data_t.shape[0]:
        raise ValueError("theta must contain one angle per trial")

    n_trials, _, n_times = data_t.shape
    relative_angles = torch.arange(
        n_templates, dtype=torch_dtype, device=target
    ) * (2 * torch.pi / n_templates)
    trialwise = torch.empty(
        (n_trials, n_times), dtype=torch_dtype, device=target
    )
    distance_sum = torch.zeros(
        (n_templates, n_times), dtype=torch_dtype, device=target
    )
    all_indices = torch.arange(n_trials)
    progress_step = max(1, n_trials // 20)

    for test_index in range(n_trials):
        train_mask = torch.ones(n_trials, dtype=torch.bool)
        train_mask[test_index] = False
        train = all_indices[train_mask].to(target)

        # MATLAB mahalTune_func uses theta(test)-angspace as each absolute
        # hard-bin centre. The grid here is [0, 2pi), but has the same circle.
        template_centres = torch.remainder(
            theta_t[test_index] - relative_angles,
            2 * torch.pi,
        )
        angle_delta = torch.remainder(
            theta_t[train][None, :] - template_centres[:, None],
            2 * torch.pi,
        )
        angle_delta = torch.minimum(angle_delta, 2 * torch.pi - angle_delta)
        selected = angle_delta < bin_half_width
        counts = selected.sum(dim=1)
        if torch.any(counts == 0):
            empty = torch.where(counts == 0)[0].cpu().tolist()
            raise ValueError(
                f"Test trial {test_index} has empty hard template(s) {empty}; "
                "increase bin width or reduce the number of templates"
            )
        weights = selected.to(data_t.dtype) / counts[:, None]
        templates = torch.einsum("kn,nft->kft", weights, data_t[train])

        covariance = _covdiag_batched(data_t[train].permute(2, 0, 1))
        trial_distances = _mahalanobis_batched(
            templates,
            data_t[test_index : test_index + 1],
            covariance,
        )[:, 0, :]
        trialwise[test_index] = -(
            torch.cos(relative_angles)[:, None] * trial_distances
        ).mean(dim=0)
        distance_sum += -(
            trial_distances - trial_distances.mean(dim=0, keepdim=True)
        )

        completed = test_index + 1
        if completed % progress_step == 0 or completed == n_trials:
            print(
                f"  hard LOOCV trials {completed}/{n_trials}",
                flush=True,
            )

    cosine = trialwise.mean(dim=0)
    distances = distance_sum / n_trials
    trialwise_np = trialwise.cpu().numpy()
    cosine_np = cosine.cpu().numpy()
    if config.smooth_ms > 0:
        from scipy.ndimage import gaussian_filter1d

        sigma_points = config.smooth_ms / config.step_ms
        cosine_np = gaussian_filter1d(cosine_np, sigma_points)
        trialwise_np = gaussian_filter1d(
            trialwise_np, sigma_points, axis=1
        )

    result = cosine_np, distances.cpu().numpy()
    if return_trialwise:
        return result + (trialwise_np,)
    return result
