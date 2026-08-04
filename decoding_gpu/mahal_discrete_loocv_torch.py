"""GPU leave-one-trial-out Mahalanobis decoding for discrete angles."""

from __future__ import annotations

import numpy as np
import torch

from helper import DecodeConfig, _covdiag_batched, _mahalanobis_batched


@torch.inference_mode()
def decode_discrete_loocv(
    features: np.ndarray,
    theta: np.ndarray,
    config: DecodeConfig,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 1,
    return_trialwise: bool = False,
    strata: np.ndarray | None = None,
):
    """Leave out each trial and decode it from balanced category templates.

    For every held-out trial, category templates are randomly subsampled to
    the smallest training-category count. Covariance uses all training trials.
    If session ``strata`` are supplied, feature scaling is fitted separately
    within each session using training trials only.
    """
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
    template_angles = torch.unique(theta_t)
    if template_angles.numel() < 2:
        raise ValueError("at least two discrete categories are required")

    strata_t = None
    if strata is not None:
        strata_t = torch.as_tensor(strata).flatten().cpu()
        if strata_t.numel() != data_t.shape[0]:
            raise ValueError("strata must contain one value per trial")

    n_trials, _, n_times = data_t.shape
    n_templates = template_angles.numel()
    trialwise = torch.empty((n_trials, n_times), dtype=torch_dtype, device=target)
    distance_sum = torch.zeros(
        (n_templates, n_times), dtype=torch_dtype, device=target
    )
    all_cpu = torch.arange(n_trials)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    progress_step = max(1, n_trials // 20)

    for test_index in range(n_trials):
        train_mask = torch.ones(n_trials, dtype=torch.bool)
        train_mask[test_index] = False
        train_cpu = all_cpu[train_mask]
        train = train_cpu.to(target)
        train_data = data_t[train].clone()
        test_data = data_t[test_index : test_index + 1].clone()

        if strata_t is not None:
            test_stratum = strata_t[test_index]
            for stratum in torch.unique(strata_t):
                train_local_cpu = torch.where(strata_t[train_cpu] == stratum)[0]
                train_local = train_local_cpu.to(target)
                mean = train_data[train_local].mean(dim=0, keepdim=True)
                sd = train_data[train_local].std(
                    dim=0, correction=0, keepdim=True
                ).clamp_min(torch.finfo(torch_dtype).eps)
                train_data[train_local] = (train_data[train_local] - mean) / sd
                if stratum == test_stratum:
                    test_data = (test_data - mean) / sd

        membership = theta_t[train][None, :] == template_angles[:, None]
        counts = membership.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError(
                f"Leaving out trial {test_index} removes a complete category"
            )
        balanced_count = int(counts.min().item())
        balanced = torch.zeros(
            membership.shape, dtype=torch_dtype, device=target
        )
        for template_index in range(n_templates):
            members_cpu = torch.where(membership[template_index].cpu())[0]
            selected_cpu = members_cpu[
                torch.randperm(members_cpu.numel(), generator=generator)[
                    :balanced_count
                ]
            ]
            balanced[template_index, selected_cpu.to(target)] = 1
        templates = torch.einsum(
            "kn,nft->kft", balanced / balanced_count, train_data
        )
        covariance = _covdiag_batched(train_data.permute(2, 0, 1))
        trial_distances = _mahalanobis_batched(
            templates, test_data, covariance
        )[:, 0, :]

        theta_delta = torch.remainder(
            theta_t[test_index] - template_angles, 2 * torch.pi
        )
        trialwise[test_index] = -(
            torch.cos(theta_delta)[:, None] * trial_distances
        ).mean(dim=0)
        distance_sum += -(
            trial_distances - trial_distances.mean(dim=0, keepdim=True)
        )

        completed = test_index + 1
        if completed % progress_step == 0 or completed == n_trials:
            print(f"  discrete LOOCV trials {completed}/{n_trials}", flush=True)

    cosine_np = trialwise.mean(dim=0).cpu().numpy()
    trialwise_np = trialwise.cpu().numpy()
    if config.smooth_ms > 0:
        from scipy.ndimage import gaussian_filter1d

        sigma_points = config.smooth_ms / config.step_ms
        cosine_np = gaussian_filter1d(cosine_np, sigma_points)
        trialwise_np = gaussian_filter1d(
            trialwise_np, sigma_points, axis=1
        )
    result = cosine_np, (distance_sum / n_trials).cpu().numpy()
    if return_trialwise:
        return result + (trialwise_np,)
    return result
