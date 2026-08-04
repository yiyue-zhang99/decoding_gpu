"""Repeated hard-bin Mahalanobis decoding with half-cosine basis templates.

This implements the continuous-orientation procedure described by Wolff and
colleagues: orientations are assigned to the nearest of 16 hard bins, folds
are stratified by those bins, training-bin counts are equalised by random
subsampling, and the bin means are convolved with a half-cosine basis raised
to a configurable exponent before Mahalanobis distances are computed.
"""

from __future__ import annotations

import numpy as np
import torch

from helper import DecodeConfig, _covdiag_batched, _mahalanobis_batched
from mahal_discrete_kfold_torch import _stratified_folds


def _nearest_orientation_bins(
    theta: torch.Tensor, centres: torch.Tensor
) -> torch.Tensor:
    """Assign doubled orientation angles to their nearest circular centre."""
    delta = torch.remainder(
        theta[:, None] - centres[None, :] + torch.pi, 2 * torch.pi
    ) - torch.pi
    return delta.abs().argmin(dim=1)


def _half_cosine_basis(
    centres: torch.Tensor, exponent: float
) -> torch.Tensor:
    """Return row-normalised half-cosine basis weights.

    ``centres`` use doubled-angle radians. Dividing the circular difference
    by two maps it back to orientation space, where differences lie between
    -pi/2 and pi/2 and the cosine is the non-negative half cosine.
    """
    delta = torch.remainder(
        centres[:, None] - centres[None, :] + torch.pi, 2 * torch.pi
    ) - torch.pi
    basis = torch.cos(delta / 2).clamp_min(0).pow(exponent)
    return basis / basis.sum(dim=1, keepdim=True)


@torch.inference_mode()
def mahal_basis_kfold(
    data: torch.Tensor,
    theta: torch.Tensor,
    n_folds: int,
    bin_centres: torch.Tensor,
    basis_exponent: float,
    generator: torch.Generator,
    strata: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one stratified K-fold draw for one shifted orientation space."""
    theta = torch.remainder(theta.flatten(), 2 * torch.pi)
    n_trials, _, n_times = data.shape
    n_bins = bin_centres.numel()
    category = _nearest_orientation_bins(theta, bin_centres)
    folds = _stratified_folds(category.cpu(), n_folds, generator, strata)
    basis = _half_cosine_basis(bin_centres, basis_exponent)
    distances = torch.full(
        (n_bins, n_trials, n_times),
        torch.nan,
        dtype=data.dtype,
        device=data.device,
    )
    all_indices = torch.arange(n_trials)

    for test_cpu in folds:
        train_mask = torch.ones(n_trials, dtype=torch.bool)
        train_mask[test_cpu] = False
        train_cpu = all_indices[train_mask]
        train = train_cpu.to(data.device)
        test = test_cpu.to(data.device)
        train_data = data[train].clone()
        test_data = data[test].clone()

        if strata is not None:
            # Pooled-session decoding: fit scaling only on training trials and
            # independently within each session, then apply it to test trials
            # from the same session.
            for stratum in torch.unique(strata):
                train_local_cpu = torch.where(strata[train_cpu] == stratum)[0]
                test_local_cpu = torch.where(strata[test_cpu] == stratum)[0]
                train_local = train_local_cpu.to(data.device)
                test_local = test_local_cpu.to(data.device)
                mean = train_data[train_local].mean(dim=0, keepdim=True)
                sd = train_data[train_local].std(
                    dim=0, correction=0, keepdim=True
                ).clamp_min(torch.finfo(data.dtype).eps)
                train_data[train_local] = (train_data[train_local] - mean) / sd
                test_data[test_local] = (test_data[test_local] - mean) / sd

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
        # Half-cosine^exponent convolution across the hard-bin means.
        templates = torch.einsum("kj,jft->kft", basis, hard_bin_means)
        covariance = _covdiag_batched(train_data.permute(2, 0, 1))
        distances[:, test, :] = _mahalanobis_batched(
            templates, test_data, covariance
        )

    theta_distance = torch.remainder(
        theta[:, None] - bin_centres[None, :] + torch.pi, 2 * torch.pi
    ) - torch.pi
    trial_cos = -(
        torch.cos(theta_distance).T[:, :, None] * distances
    ).mean(dim=0)
    # Mean-centre/sign-reverse each trial's profile, then order the bins by
    # angular difference from that trial's true orientation. This is the
    # similarity profile described in the method and makes profiles from the
    # eight shifted orientation spaces directly averageable.
    similarity = -(distances - distances.mean(dim=0, keepdim=True))
    relative_order = theta_distance.argsort(dim=1)
    similarity_by_trial = similarity.permute(1, 0, 2)
    aligned = torch.gather(
        similarity_by_trial,
        1,
        relative_order[:, :, None].expand(-1, -1, n_times),
    )
    return trial_cos, aligned.permute(1, 0, 2)


def decode_basis_kfold_repetitions(
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
):
    """Decode continuous orientations over shifted hard-bin spaces.

    With 16 bins, adjacent centres are 11.25 orientation degrees apart and
    nearest-centre assignment gives a hard-bin half-width of 5.625 degrees.
    Eight spaces shift the grid by 1.40625 degrees per space, matching the
    eight grids described in the method.
    """
    if n_bins < 4:
        raise ValueError("n_bins must be at least 4")
    if n_orientation_spaces < 1:
        raise ValueError("n_orientation_spaces must be positive")
    if basis_exponent <= 0 or repetitions < 1:
        raise ValueError("basis_exponent and repetitions must be positive")
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
    trial_sum = torch.zeros(
        (data_t.shape[0], data_t.shape[2]), dtype=torch_dtype, device=target
    )
    distance_sum = torch.zeros(
        (n_bins, data_t.shape[2]), dtype=torch_dtype, device=target
    )
    base_centres = torch.arange(
        n_bins, dtype=torch_dtype, device=target
    ) * (2 * torch.pi / n_bins)
    total_runs = repetitions * n_orientation_spaces
    completed = 0
    for space_index in range(n_orientation_spaces):
        # One eighth of the 11.25-degree bin spacing per shifted space.
        shift = space_index * (2 * torch.pi / n_bins / n_orientation_spaces)
        centres = torch.remainder(base_centres + shift, 2 * torch.pi)
        for repetition in range(repetitions):
            trial_cos, distances = mahal_basis_kfold(
                data_t,
                theta_t,
                config.n_folds,
                centres,
                basis_exponent,
                generator,
                strata_t,
            )
            trial_sum += trial_cos
            distance_sum += distances.mean(dim=1)
            completed += 1
            print(
                f"  basis k-fold space {space_index + 1}/{n_orientation_spaces}, "
                f"repetition {repetition + 1}/{repetitions} "
                f"({completed}/{total_runs})",
                flush=True,
            )

    trialwise = trial_sum / total_runs
    cosine = trialwise.mean(dim=0)
    cosine_np = cosine.cpu().numpy()
    trialwise_np = trialwise.cpu().numpy()
    if config.smooth_ms > 0:
        from scipy.ndimage import gaussian_filter1d

        sigma_points = config.smooth_ms / config.step_ms
        cosine_np = gaussian_filter1d(cosine_np, sigma_points)
        trialwise_np = gaussian_filter1d(
            trialwise_np, sigma_points, axis=1
        )
    result = cosine_np, (distance_sum / total_runs).cpu().numpy()
    if return_trialwise:
        return result + (trialwise_np,)
    return result
