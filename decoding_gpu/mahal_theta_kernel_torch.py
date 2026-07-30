"""GPU continuous-angle kernel-smoothed Mahalanobis decoding."""

from __future__ import annotations

import numpy as np
import torch

from mahal_theta_torch import (
    DecodeConfig,
    _covdiag_batched,
    _mahalanobis_batched,
)


def _random_folds(
    n_trials: int,
    n_folds: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Split trials into similarly sized folds without discretising angles."""
    if n_trials < n_folds:
        raise ValueError(f"{n_trials} trials cannot support {n_folds} folds")
    order = torch.randperm(n_trials, generator=generator)
    return [part.sort().values for part in torch.tensor_split(order, n_folds)]


@torch.inference_mode()
def mahal_theta_kernel_kfold(
    data: torch.Tensor,
    theta: torch.Tensor,
    n_folds: int,
    template_angles: torch.Tensor,
    kernel_sigma: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode continuous angles using circular-Gaussian training templates.

    Parameters
    ----------
    data
        Trial x feature x time.
    theta
        One continuous doubled-angle value per trial, in radians.
    template_angles
        Evaluation grid around the full doubled-angle circle. Trials are not
        assigned to these points; every training trial contributes softly.
    kernel_sigma
        Circular Gaussian standard deviation in doubled-angle radians.
    """
    if kernel_sigma <= 0:
        raise ValueError("kernel_sigma must be positive")
    theta = torch.remainder(theta.flatten(), 2 * torch.pi)
    n_trials, _, n_times = data.shape
    n_templates = template_angles.numel()
    distances = torch.full(
        (n_templates, n_trials, n_times),
        torch.nan,
        dtype=data.dtype,
        device=data.device,
    )
    folds = _random_folds(n_trials, n_folds, generator)
    all_indices = torch.arange(n_trials)

    for test_cpu in folds:
        train_mask = torch.ones(n_trials, dtype=torch.bool)
        train_mask[test_cpu] = False
        train_cpu = all_indices[train_mask]
        train = train_cpu.to(data.device)
        test = test_cpu.to(data.device)

        # template x training-trial weights. No hard angle/bin assignment.
        angle_delta = torch.remainder(
            theta[train][None, :] - template_angles[:, None],
            2 * torch.pi,
        )
        angle_delta = torch.minimum(angle_delta, 2 * torch.pi - angle_delta)
        weights = torch.exp(-0.5 * (angle_delta / kernel_sigma).square())
        weight_sum = weights.sum(dim=1)
        if torch.any(weight_sum <= torch.finfo(data.dtype).eps):
            raise ValueError(
                "Kernel width is too narrow to construct every template"
            )
        weights = weights / weight_sum[:, None]
        # Weighted average of all training trials at every feature and time.
        templates = torch.einsum("kn,nft->kft", weights, data[train])

        covariance = _covdiag_batched(data[train].permute(2, 0, 1))
        distances[:, test, :] = _mahalanobis_batched(
            templates, data[test], covariance
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


def decode_kernel_repetitions(
    features: np.ndarray,
    theta: np.ndarray,
    config: DecodeConfig,
    kernel_sigma: float,
    n_templates: int = 16,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 1,
    return_trialwise: bool = False,
):
    """Run repeated continuous-angle kernel Mahalanobis decoding."""
    if n_templates < 4:
        raise ValueError("n_templates must be at least 4")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch cannot access a CUDA device")
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    target = torch.device(device)
    data_t = torch.as_tensor(features, dtype=torch_dtype, device=target)
    theta_t = torch.as_tensor(theta, dtype=torch_dtype, device=target).flatten()
    template_angles = torch.arange(
        n_templates, dtype=torch_dtype, device=target
    ) * (2 * torch.pi / n_templates)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    trial_sum = torch.zeros(
        (data_t.shape[0], data_t.shape[2]), dtype=torch_dtype, device=target
    )
    distance_sum = torch.zeros(
        (n_templates, data_t.shape[2]), dtype=torch_dtype, device=target
    )
    for repetition in range(config.reps):
        trial_cos, distances = mahal_theta_kernel_kfold(
            data_t,
            theta_t,
            config.n_folds,
            template_angles,
            kernel_sigma,
            generator,
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
