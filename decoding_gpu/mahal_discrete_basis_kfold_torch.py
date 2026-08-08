"""GPU discrete-category k-fold Mahalanobis decoding.

For designs with a small number of exact orientation categories (e.g. guven's
6 cue/uncue item categories) rather than a continuous angle. Folds are
stratified by category, and within every fold the training trials used for
each template are randomly subsampled to the smallest category count. The
public decoders provide either the resulting hard category means directly or
convolve them with the reference half-cosine basis before distance estimation.
"""

from __future__ import annotations

import numpy as np
import torch

from helper import (
    DecodeConfig,
    _covdiag_batched,
    _mahalanobis_batched,
)


def _stratified_folds(
    theta: torch.Tensor,
    n_folds: int,
    generator: torch.Generator,
    strata: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Split trials into folds, drawing each category/stratum separately.

    Every fold has close to 1/n_folds of every category, so the training set
    for any held-out fold always retains trials from all categories.  When
    ``strata`` is supplied (for example, session IDs), splitting is performed
    separately within every stratum x category cell.
    """
    n_trials = theta.numel()
    if strata is None:
        strata = torch.zeros(n_trials, dtype=torch.long)
    else:
        strata = strata.flatten().cpu()
        if strata.numel() != n_trials:
            raise ValueError("strata must contain one value per trial")
    fold_of = torch.empty(n_trials, dtype=torch.long)
    for stratum in torch.unique(strata):
        in_stratum = strata == stratum
        for value in torch.unique(theta):
            members = torch.where(in_stratum & (theta == value))[0]
            if members.numel() < n_folds:
                raise ValueError(
                    f"stratum {stratum.item()}, category {value.item():.4f} "
                    f"has {members.numel()} trials, fewer than "
                    f"n_folds={n_folds}"
                )
            order = members[torch.randperm(members.numel(), generator=generator)]
            for fold_index, part in enumerate(torch.tensor_split(order, n_folds)):
                fold_of[part] = fold_index
    return [
        torch.where(fold_of == fold_index)[0].sort().values
        for fold_index in range(n_folds)
    ]


@torch.inference_mode()
def mahal_discrete_kfold(
    data: torch.Tensor,
    theta: torch.Tensor,
    n_folds: int,
    template_angles: torch.Tensor,
    generator: torch.Generator,
    strata: torch.Tensor | None = None,
    basis_exponent: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode exact discrete angle categories with hard k-fold templates.

    Parameters
    ----------
    data
        Trial x feature x time.
    theta
        One doubled-angle value per trial, in radians. Must take on exactly
        the values in `template_angles` (bit-identical per category).
    template_angles
        The full set of realised category angles; every training trial is
        assigned to exactly one of these (equality, not distance/kernel).
    """
    theta = torch.remainder(theta.flatten(), 2 * torch.pi)
    n_trials, _, n_times = data.shape
    n_templates = template_angles.numel()
    distances = torch.full(
        (n_templates, n_trials, n_times),
        torch.nan,
        dtype=data.dtype,
        device=data.device,
    )
    folds = _stratified_folds(theta.cpu(), n_folds, generator, strata)
    basis = None
    if basis_exponent is not None:
        if basis_exponent <= 0:
            raise ValueError("basis_exponent must be positive")
        delta = torch.remainder(
            template_angles[:, None] - template_angles[None, :] + torch.pi,
            2 * torch.pi,
        ) - torch.pi
        # Exact basis used by mahal_theta_kfold_basis_b.m:
        # (0.5 + 0.5*cos(theta-mu))^(number of categories - 1).
        basis = (0.5 + 0.5 * torch.cos(delta)).pow(basis_exponent)
        basis = basis / basis.sum(dim=1, keepdim=True)
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
            # Fit feature scaling on training trials only, independently for
            # each session/stratum, then apply it to that stratum's test data.
            # Shapes are trial x feature x time.
            for stratum in torch.unique(strata):
                train_local_cpu = torch.where(strata[train_cpu] == stratum)[0]
                test_local_cpu = torch.where(strata[test_cpu] == stratum)[0]
                train_local = train_local_cpu.to(data.device)
                test_local = test_local_cpu.to(data.device)
                mean = train_data[train_local].mean(dim=0, keepdim=True)
                sd = train_data[train_local].std(
                    dim=0, correction=0, keepdim=True
                )
                sd = sd.clamp_min(torch.finfo(data.dtype).eps)
                train_data[train_local] = (train_data[train_local] - mean) / sd
                test_data[test_local] = (test_data[test_local] - mean) / sd

        # Template x training-trial hard membership (exact category match).
        # Match mahal_func_theta_kfold_b.m: randomly subsample every category
        # to the smallest training-category count before averaging templates.
        # Covariance below deliberately continues to use all training trials.
        membership = (
            theta[train][None, :] == template_angles[:, None]
        ).to(data.dtype)
        counts = membership.sum(dim=1).to(torch.long)
        if torch.any(counts == 0):
            empty = torch.where(counts == 0)[0].cpu().tolist()
            raise ValueError(
                f"Fold's training set is missing categor{'y' if len(empty) == 1 else 'ies'} "
                f"{empty}; reduce n_folds or check category balance"
            )
        balanced_count = int(counts.min().item())
        balanced_membership = torch.zeros_like(membership)
        for template_index in range(n_templates):
            members_cpu = torch.where(
                membership[template_index].cpu().bool()
            )[0]
            selected_cpu = members_cpu[
                torch.randperm(members_cpu.numel(), generator=generator)[
                    :balanced_count
                ]
            ]
            balanced_membership[
                template_index, selected_cpu.to(data.device)
            ] = 1
        weights = balanced_membership / balanced_count
        hard_templates = torch.einsum("kn,nft->kft", weights, train_data)
        templates = (
            hard_templates
            if basis is None
            else torch.einsum("kj,jft->kft", basis, hard_templates)
        )

        covariance = _covdiag_batched(train_data.permute(2, 0, 1))
        distances[:, test, :] = _mahalanobis_batched(
            templates, test_data, covariance
        )

    theta_distance = torch.remainder(
        theta[:, None] - template_angles[None, :],
        2 * torch.pi,
    )
    trial_cos = -(
        torch.cos(theta_distance).T[:, :, None] * distances
    ).mean(dim=0)
    distances = -(distances - distances.mean(dim=0, keepdim=True))
    return trial_cos, distances


def _decode_discrete_repetitions(
    features: np.ndarray,
    theta: np.ndarray,
    config: DecodeConfig,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 1,
    return_trialwise: bool = False,
    strata: np.ndarray | None = None,
    basis_exponent: float | None = None,
):
    """Shared repeated hard/basis discrete-category decoder."""
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
    template_angles = torch.unique(theta_t)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    strata_t = None
    if strata is not None:
        strata_t = torch.as_tensor(strata).flatten().cpu()
        if strata_t.numel() != data_t.shape[0]:
            raise ValueError("strata must contain one value per trial")

    trial_sum = torch.zeros(
        (data_t.shape[0], data_t.shape[2]), dtype=torch_dtype, device=target
    )
    distance_sum = torch.zeros(
        (template_angles.numel(), data_t.shape[2]), dtype=torch_dtype, device=target
    )
    for repetition in range(config.reps):
        trial_cos, distances = mahal_discrete_kfold(
            data_t,
            theta_t,
            config.n_folds,
            template_angles,
            generator,
            strata_t,
            basis_exponent,
        )
        trial_sum += trial_cos
        distance_sum += distances.mean(dim=1)
        print(f"  repetition {repetition + 1}/{config.reps}", flush=True)

    trialwise = trial_sum / config.reps
    cosine = trialwise.mean(dim=0)
    if config.smooth_ms > 0:
        from scipy.ndimage import gaussian_filter1d

        sigma_points = config.smooth_ms / config.step_ms
        cosine_np = gaussian_filter1d(cosine.cpu().numpy(), sigma_points)
        trialwise_np = gaussian_filter1d(
            trialwise.cpu().numpy(), sigma_points, axis=1
        )
    else:
        cosine_np = cosine.cpu().numpy()
        trialwise_np = trialwise.cpu().numpy()
    result = (
        cosine_np,
        (distance_sum / config.reps).cpu().numpy(),
    )
    if return_trialwise:
        return result + (trialwise_np,)
    return result


def decode_discrete_repetitions(
    features: np.ndarray,
    theta: np.ndarray,
    config: DecodeConfig,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 1,
    return_trialwise: bool = False,
    strata: np.ndarray | None = None,
):
    """Run repeated discrete-category k-fold decoding with hard templates."""
    return _decode_discrete_repetitions(
        features,
        theta,
        config,
        device=device,
        dtype=dtype,
        seed=seed,
        return_trialwise=return_trialwise,
        strata=strata,
        basis_exponent=None,
    )


def decode_discrete_basis_repetitions(
    features: np.ndarray,
    theta: np.ndarray,
    config: DecodeConfig,
    basis_exponent: float = 5.0,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 1,
    return_trialwise: bool = False,
    strata: np.ndarray | None = None,
):
    """Run repeated discrete k-fold decoding with half-cosine basis templates.

    Training trials are first randomly subsampled so every exact category has
    the same count. Category means are then convolved with the row-normalised
    ``(0.5 + 0.5*cos(theta-mu))**basis_exponent`` basis before distances are
    computed. For Guven's six categories the reference exponent is five.
    """
    return _decode_discrete_repetitions(
        features,
        theta,
        config,
        device=device,
        dtype=dtype,
        seed=seed,
        return_trialwise=return_trialwise,
        strata=strata,
        basis_exponent=basis_exponent,
    )
