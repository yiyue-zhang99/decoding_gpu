"""Time-cross-generalized discrete-category Mahalanobis decoding.

Rows of every output time matrix are training times and columns are test
times. Templates and covariance matrices are estimated at the training time,
then applied to every test time, following ``mahalTune_func_cross_temp.m``.
"""

from __future__ import annotations

import numpy as np
import torch

from helper import DecodeConfig, _covdiag_batched
from mahal_discrete_basis_kfold_torch import _stratified_folds


def _mahalanobis_time_cross(
    templates: torch.Tensor,
    tests: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    """Return template x test-trial x train-time x test-time distances."""
    templates_t = templates.permute(2, 0, 1)
    tests_t = tests.permute(2, 0, 1)
    n_train_times, n_templates, n_features = templates_t.shape
    n_test_times, n_tests, _ = tests_t.shape

    chol, info = torch.linalg.cholesky_ex(covariance)
    if torch.any(info):
        eps = torch.finfo(covariance.dtype).eps
        scale = torch.diagonal(covariance, dim1=-2, dim2=-1).mean(dim=-1)
        jitter = (100 * eps * scale.clamp_min(eps))[:, None, None]
        covariance = covariance + jitter * torch.eye(
            n_features, dtype=covariance.dtype, device=covariance.device
        )
        chol = torch.linalg.cholesky(covariance)

    distances = torch.empty(
        (n_templates, n_tests, n_train_times, n_test_times),
        dtype=templates.dtype,
        device=templates.device,
    )
    # Process one training time at a time. This avoids materialising the full
    # train-time x test-time x template x trial x feature difference tensor.
    for train_time in range(n_train_times):
        difference = (
            templates_t[train_time][None, :, None, :]
            - tests_t[:, None, :, :]
        )
        rhs = difference.permute(3, 0, 1, 2).reshape(n_features, -1)
        whitened = torch.linalg.solve_triangular(
            chol[train_time], rhs, upper=False
        )
        distances[:, :, train_time, :] = (
            whitened.square()
            .sum(dim=0)
            .sqrt()
            .reshape(n_test_times, n_templates, n_tests)
            .permute(1, 2, 0)
        )
    return distances


def _standardize_by_stratum(
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    train_cpu: torch.Tensor,
    test_cpu: torch.Tensor,
    strata: torch.Tensor | None,
) -> None:
    """Apply fold-safe, per-stratum scaling in place."""
    if strata is None:
        return
    for stratum in torch.unique(strata):
        train_local = torch.where(strata[train_cpu] == stratum)[0].to(
            train_data.device
        )
        test_local = torch.where(strata[test_cpu] == stratum)[0].to(
            test_data.device
        )
        mean = train_data[train_local].mean(dim=0, keepdim=True)
        sd = train_data[train_local].std(
            dim=0, correction=0, keepdim=True
        ).clamp_min(torch.finfo(train_data.dtype).eps)
        train_data[train_local] = (train_data[train_local] - mean) / sd
        test_data[test_local] = (test_data[test_local] - mean) / sd


@torch.inference_mode()
def mahal_discrete_kfold_time_cross_generalization(
    data: torch.Tensor,
    theta: torch.Tensor,
    n_folds: int,
    template_angles: torch.Tensor,
    generator: torch.Generator,
    strata: torch.Tensor | None = None,
    basis_exponent: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one discrete-category K-fold time-generalization draw.

    Returns
    -------
    trial_cos
        Trial x train-time x test-time cosine-weighted decoding values.
    distances
        Template x trial x train-time x test-time similarity profiles.
    """
    theta = torch.remainder(theta.flatten(), 2 * torch.pi)
    n_trials, _, n_times = data.shape
    n_templates = template_angles.numel()
    distances = torch.full(
        (n_templates, n_trials, n_times, n_times),
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
        _standardize_by_stratum(
            train_data, test_data, train_cpu, test_cpu, strata
        )

        membership = (
            theta[train][None, :] == template_angles[:, None]
        ).to(data.dtype)
        counts = membership.sum(dim=1).to(torch.long)
        if torch.any(counts == 0):
            empty = torch.where(counts == 0)[0].cpu().tolist()
            raise ValueError(
                f"Fold training data are missing category/categories {empty}; "
                "reduce n_folds or check category balance"
            )
        balanced_count = int(counts.min().item())
        balanced_membership = torch.zeros_like(membership)
        for template_index in range(n_templates):
            members_cpu = torch.where(membership[template_index].cpu().bool())[0]
            selected_cpu = members_cpu[
                torch.randperm(members_cpu.numel(), generator=generator)[
                    :balanced_count
                ]
            ]
            balanced_membership[
                template_index, selected_cpu.to(data.device)
            ] = 1

        hard_templates = torch.einsum(
            "kn,nft->kft", balanced_membership / balanced_count, train_data
        )
        templates = (
            hard_templates
            if basis is None
            else torch.einsum("kj,jft->kft", basis, hard_templates)
        )
        covariance = _covdiag_batched(train_data.permute(2, 0, 1))
        distances[:, test, :, :] = _mahalanobis_time_cross(
            templates, test_data, covariance
        )

    theta_distance = torch.remainder(
        theta[:, None] - template_angles[None, :], 2 * torch.pi
    )
    trial_cos = -(
        torch.cos(theta_distance).T[:, :, None, None] * distances
    ).mean(dim=0)
    similarity = -(distances - distances.mean(dim=0, keepdim=True))
    return trial_cos, similarity


def _decode_discrete_time_cross_repetitions(
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
    """Shared repeated hard/basis discrete time-generalization decoder."""
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
    template_angles = torch.unique(theta_t)
    strata_t = None
    if strata is not None:
        strata_t = torch.as_tensor(strata).flatten().cpu()
        if strata_t.numel() != data_t.shape[0]:
            raise ValueError("strata must contain one value per trial")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (data_t.shape[0], data_t.shape[2], data_t.shape[2])
    trial_sum = torch.zeros(shape, dtype=torch_dtype, device=target)
    distance_sum = torch.zeros(
        (template_angles.numel(), data_t.shape[2], data_t.shape[2]),
        dtype=torch_dtype,
        device=target,
    )
    for repetition in range(config.reps):
        trial_cos, distances = mahal_discrete_kfold_time_cross_generalization(
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
        print(
            f"  time-cross repetition {repetition + 1}/{config.reps}",
            flush=True,
        )

    trialwise = trial_sum / config.reps
    cosine = trialwise.mean(dim=0)
    cosine_np = cosine.cpu().numpy()
    trialwise_np = trialwise.cpu().numpy()
    if config.smooth_ms > 0:
        from scipy.ndimage import gaussian_filter

        sigma = config.smooth_ms / config.step_ms
        cosine_np = gaussian_filter(cosine_np, sigma=(sigma, sigma))
        trialwise_np = gaussian_filter(
            trialwise_np, sigma=(0, sigma, sigma)
        )
    result = cosine_np, (distance_sum / config.reps).cpu().numpy()
    if return_trialwise:
        return result + (trialwise_np,)
    return result


def decode_discrete_time_cross_repetitions(
    features: np.ndarray,
    theta: np.ndarray,
    config: DecodeConfig,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 1,
    return_trialwise: bool = False,
    strata: np.ndarray | None = None,
):
    """Repeated hard-template discrete time-generalization decoding."""
    return _decode_discrete_time_cross_repetitions(
        features, theta, config, device, dtype, seed, return_trialwise, strata
    )


def decode_discrete_basis_time_cross_repetitions(
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
    """Repeated basis-template discrete time-generalization decoding."""
    return _decode_discrete_time_cross_repetitions(
        features,
        theta,
        config,
        device,
        dtype,
        seed,
        return_trialwise,
        strata,
        basis_exponent,
    )
