"""Time-cross-generalized continuous-orientation basis decoding."""

from __future__ import annotations

import numpy as np
import torch

from helper import DecodeConfig, _covdiag_batched
from mahal_discrete_basis_kfold_torch import _stratified_folds
from mahal_discrete_basis_kfold_torch_time_cross_generalization import (
    _mahalanobis_time_cross,
    _standardize_by_stratum,
)
from mahal_theta_basis_kfold_torch import (
    _half_cosine_basis,
    _nearest_orientation_bins,
)


@torch.inference_mode()
def mahal_basis_kfold_time_cross_generalization(
    data: torch.Tensor,
    theta: torch.Tensor,
    n_folds: int,
    bin_centres: torch.Tensor,
    basis_exponent: float,
    generator: torch.Generator,
    strata: torch.Tensor | None = None,
    train_time_batch: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one shifted-space K-fold draw over all train/test times.

    Training times are processed in batches so original-resolution time-cross
    decoding does not materialise bin x trial x train-time x test-time on the
    GPU. The returned profile is already averaged over held-out trials.
    """
    theta = torch.remainder(theta.flatten(), 2 * torch.pi)
    n_trials, _, n_times = data.shape
    n_bins = bin_centres.numel()
    category = _nearest_orientation_bins(theta, bin_centres)
    folds = _stratified_folds(category.cpu(), n_folds, generator, strata)
    basis = _half_cosine_basis(bin_centres, basis_exponent)
    if train_time_batch < 1:
        raise ValueError("train_time_batch must be positive")
    trial_cos = torch.empty(
        (n_trials, n_times, n_times),
        dtype=data.dtype,
        device=data.device,
    )
    aligned_sum = torch.zeros(
        (n_bins, n_times, n_times), dtype=data.dtype, device=data.device
    )
    all_indices = torch.arange(n_trials)
    theta_distance = torch.remainder(
        theta[:, None] - bin_centres[None, :] + torch.pi, 2 * torch.pi
    ) - torch.pi
    relative_order = theta_distance.argsort(dim=1)

    for test_cpu in folds:
        train_mask = torch.ones(n_trials, dtype=torch.bool)
        train_mask[test_cpu] = False
        train_cpu = all_indices[train_mask]
        train = train_cpu.to(data.device)
        test = test_cpu.to(data.device)
        train_data = data[train].clone()
        test_data = data[test].clone()
        _standardize_by_stratum(
            train_data, test_data, train_cpu, test_cpu, strata
        )

        membership = (
            category[train][None, :]
            == torch.arange(n_bins, device=data.device)[:, None]
        )
        counts = membership.sum(dim=1)
        if torch.any(counts == 0):
            empty = torch.where(counts == 0)[0].cpu().tolist()
            raise ValueError(
                f"Fold training data are missing hard bin(s) {empty}; "
                "reduce n_folds or check the orientation distribution"
            )
        balanced_count = int(counts.min().item())
        balanced_membership = torch.zeros(
            membership.shape, dtype=data.dtype, device=data.device
        )
        for bin_index in range(n_bins):
            members_cpu = torch.where(membership[bin_index].cpu())[0]
            selected_cpu = members_cpu[
                torch.randperm(members_cpu.numel(), generator=generator)[
                    :balanced_count
                ]
            ]
            balanced_membership[
                bin_index, selected_cpu.to(data.device)
            ] = 1

        hard_bin_means = torch.einsum(
            "kn,nft->kft", balanced_membership / balanced_count, train_data
        )
        templates = torch.einsum("kj,jft->kft", basis, hard_bin_means)
        covariance = _covdiag_batched(train_data.permute(2, 0, 1))
        weights = torch.cos(theta_distance[test]).T
        fold_order = relative_order[test]
        for first in range(0, n_times, train_time_batch):
            stop = min(first + train_time_batch, n_times)
            distances = _mahalanobis_time_cross(
                templates[:, :, first:stop],
                test_data,
                covariance[first:stop],
            )
            trial_cos[test, first:stop, :] = -(
                weights[:, :, None, None] * distances
            ).mean(dim=0)
            similarity = -(distances - distances.mean(dim=0, keepdim=True))
            aligned = torch.gather(
                similarity.permute(1, 0, 2, 3),
                1,
                fold_order[:, :, None, None].expand(
                    -1, -1, stop - first, n_times
                ),
            )
            aligned_sum[:, first:stop, :] += aligned.sum(dim=0)

    return trial_cos, aligned_sum / n_trials


def decode_basis_kfold_time_cross_repetitions(
    features: np.ndarray,
    theta: np.ndarray,
    config: DecodeConfig,
    n_bins: int = 16,
    n_orientation_spaces: int = 8,
    basis_exponent: float = 15.0,
    repetitions: int = 100,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 1,
    return_trialwise: bool = False,
    strata: np.ndarray | None = None,
    train_time_batch: int = 16,
):
    """Decode shifted orientation spaces into train-time x test-time maps."""
    if n_bins < 4:
        raise ValueError("n_bins must be at least 4")
    if n_orientation_spaces < 1:
        raise ValueError("n_orientation_spaces must be positive")
    if basis_exponent <= 0 or repetitions < 1:
        raise ValueError("basis_exponent and repetitions must be positive")
    if train_time_batch < 1:
        raise ValueError("train_time_batch must be positive")
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
    if data_t.ndim != 3 or theta_t.numel() != data_t.shape[0]:
        raise ValueError("features must be trial x feature x time with one theta per trial")
    strata_t = None
    if strata is not None:
        strata_t = torch.as_tensor(strata).flatten().cpu()
        if strata_t.numel() != data_t.shape[0]:
            raise ValueError("strata must contain one value per trial")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n_times = data_t.shape[2]
    trial_sum = torch.zeros(
        (data_t.shape[0], n_times, n_times),
        dtype=torch_dtype,
        device=target,
    )
    distance_sum = torch.zeros(
        (n_bins, n_times, n_times), dtype=torch_dtype, device=target
    )
    base_centres = torch.arange(
        n_bins, dtype=torch_dtype, device=target
    ) * (2 * torch.pi / n_bins)
    total_runs = repetitions * n_orientation_spaces
    completed = 0
    for space_index in range(n_orientation_spaces):
        shift = space_index * (2 * torch.pi / n_bins / n_orientation_spaces)
        centres = torch.remainder(base_centres + shift, 2 * torch.pi)
        for repetition in range(repetitions):
            trial_cos, mean_distances = mahal_basis_kfold_time_cross_generalization(
                data_t,
                theta_t,
                config.n_folds,
                centres,
                basis_exponent,
                generator,
                strata_t,
                train_time_batch=train_time_batch,
            )
            trial_sum += trial_cos
            distance_sum += mean_distances
            completed += 1
            print(
                f"  basis time-cross space {space_index + 1}/"
                f"{n_orientation_spaces}, repetition {repetition + 1}/"
                f"{repetitions} ({completed}/{total_runs})",
                flush=True,
            )

    trialwise = trial_sum / total_runs
    cosine_np = trialwise.mean(dim=0).cpu().numpy()
    trialwise_np = trialwise.cpu().numpy()
    if config.smooth_ms > 0:
        from scipy.ndimage import gaussian_filter

        sigma = config.smooth_ms / config.step_ms
        cosine_np = gaussian_filter(cosine_np, sigma=(sigma, sigma))
        trialwise_np = gaussian_filter(
            trialwise_np, sigma=(0, sigma, sigma)
        )
    result = cosine_np, (distance_sum / total_runs).cpu().numpy()
    if return_trialwise:
        return result + (trialwise_np,)
    return result
