"""Batched PyTorch implementation of the Wolff k-fold Mahalanobis decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


#定义param
@dataclass(frozen=True)
class DecodeConfig:
    n_folds: int = 8
    reps: int = 50
    step_ms: float = 10.0
    span_ms: float = 10.0
    window_ms: float = 100.0
    smooth_ms: float = 0.0
    toi: tuple[float, float] = (-0.05, 1.2)


def make_sliding_features(
    data: np.ndarray,
    time: np.ndarray,
    config: DecodeConfig,
    mode: str = "sliding-window",
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window or single-timepoint decoding features."""
    if data.ndim != 3:
        raise ValueError("data must have shape trial x channel x time")
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if data.shape[2] != time.size:
        raise ValueError("data time dimension does not match the time vector")
    if mode not in {"sliding-window", "timepoint"}:
        raise ValueError(
            "mode must be 'sliding-window' or 'timepoint'; "
            f"got {mode!r}"
        )

    hz = round(1.0 / np.median(np.diff(time)))
    eligible = np.flatnonzero((time >= config.toi[0]) & (time <= config.toi[1])) #生成bool
    if eligible.size == 0:
        raise ValueError("No samples fall inside the requested time range")

    requested = np.arange(
        time[eligible[0]],
        time[eligible[-1]] + config.step_ms / 2000,
        config.step_ms / 1000,
    ) #每次间隔step_ms取点1
    indices = np.array([np.argmin(np.abs(time - value)) for value in requested]) #找出切片点所在的
    indices = np.unique(indices) #decoding过后保留的时间点
    if mode == "timepoint":
        return np.asarray(data[:, :, indices]), time[indices]

    window_samples = round(config.window_ms * hz / 1000)
    span_samples = round(config.span_ms * hz / 1000)
    if window_samples < 1 or span_samples < 1:
        raise ValueError("window_ms and span_ms must contain at least one sample")
    if window_samples % span_samples:
        raise ValueError("window_ms must contain an integer number of spans")
    samples_before = window_samples // 2
    samples_after = window_samples - samples_before
    indices = indices[
        (indices >= samples_before)
        & (indices + samples_after <= time.size)
    ]
    if indices.size == 0:
        raise ValueError(
            "No requested decoding times have a complete centred window"
        )

    n_trials, n_channels, _ = data.shape
    n_segments = window_samples // span_samples
    features = np.empty(
        (n_trials, n_channels * n_segments, indices.size), dtype=data.dtype
    )

    for output_index, sample_index in enumerate(indices):
        window = data[
            :, :, sample_index - samples_before : sample_index + samples_after
        ]
        window = window - window.mean(axis=2, keepdims=True)
        window = window.reshape(n_trials, n_channels, n_segments, span_samples)
        window = window.mean(axis=3)
        features[:, :, output_index] = window.reshape(
            n_trials, n_channels * n_segments
        )
    return features, time[indices]


def _split_indices_by_label(
    labels: torch.Tensor, #每个trial的类别（属于哪个bin）
    n_folds: int, 
    generator: torch.Generator, #随机数生成
) -> list[torch.Tensor]:
    fold_members: list[list[torch.Tensor]] = [[] for _ in range(n_folds)]
    for template in torch.unique(labels, sorted=True):
        members = torch.where(labels == template)[0]
        order = torch.randperm(members.numel(), generator=generator)
        members = members[order]
        for fold in range(n_folds):
            fold_members[fold].append(members[fold::n_folds]) #没有间隔n_folds取出一个trail
    return [torch.cat(parts).sort().values for parts in fold_members]


def _covdiag_batched(x: torch.Tensor) -> torch.Tensor:
    """Ledoit-Wolf diagonal-target covariance for time x trial x feature."""
    n_trials, n_features = x.shape[1:]
    x = x - x.mean(dim=1, keepdim=True)
    sample = torch.bmm(x.transpose(1, 2), x) / n_trials
    prior = torch.diag_embed(torch.diagonal(sample, dim1=-2, dim2=-1))
    d = (sample - prior).square().sum(dim=(-2, -1)) / n_features
    y = x.square()
    yty_sum = torch.bmm(y.transpose(1, 2), y).sum(dim=(-2, -1))
    r2 = (
        yty_sum / (n_features * n_trials**2)
        - sample.square().sum(dim=(-2, -1)) / (n_features * n_trials)
    )
    shrinkage = torch.where(
        d > 0, (r2 / d).clamp(0, 1), torch.ones_like(d)
    )
    return (
        shrinkage[:, None, None] * prior
        + (1 - shrinkage[:, None, None]) * sample
    )


def _mahalanobis_batched(
    templates: torch.Tensor, #trail * feature * time
    tests: torch.Tensor,
    covariance: torch.Tensor, #time * feature * feature
) -> torch.Tensor: #templates, test_trial, time_count
    """Return template x test x time distances without matrix inversion."""
    # Inputs become time x template/test x feature.
    templates_t = templates.permute(2, 0, 1)
    tests_t = tests.permute(2, 0, 1)
    difference = templates_t[:, :, None, :] - tests_t[:, None, :, :]
    time_count, template_count, test_count, feature_count = difference.shape
    rhs = difference.reshape(time_count, template_count * test_count, feature_count) #time * (template * test) * feature

    chol, info = torch.linalg.cholesky_ex(covariance) #info = 0 分解成功； info > 0, 分解失败
    if torch.any(info): #如果有时间点没有分解成功，就执行修复
        eps = torch.finfo(covariance.dtype).eps
        scale = torch.diagonal(covariance, dim1=-2, dim2=-1).mean(dim=-1)
        jitter = (100 * eps * scale.clamp_min(eps))[:, None, None]
        covariance = covariance + jitter * torch.eye(
            feature_count, dtype=covariance.dtype, device=covariance.device
        )
        chol = torch.linalg.cholesky(covariance)

    whitened = torch.linalg.solve_triangular(
        chol, rhs.transpose(1, 2), upper=False
    ) #z: (time, feature, template*test) 
    distances = whitened.square().sum(dim=1).sqrt() #(time, template, test)
    return distances.reshape(
        time_count, template_count, test_count
    ).permute(1, 2, 0)


@torch.inference_mode()
def mahal_theta_kfold(
    data: torch.Tensor,
    theta: torch.Tensor, #radians
    n_folds: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one repetition; data is trial x feature x time."""
    theta = torch.remainder(theta.flatten(),2 * torch.pi)
    template_angles, labels = torch.unique(theta, sorted=True, return_inverse=True) 
    n_trials, _, n_times = data.shape
    counts = torch.bincount(labels, minlength=template_angles.numel()) #总data每个条件的数量
    if int(counts.min().item()) < n_folds:
        raise ValueError(
            f"Each orientation template needs at least {n_folds} trials for "
            f"{n_folds}-fold decoding, but the minimum is "
            f"{int(counts.min().item())}. Bin continuous orientations before "
            "calling the decoder."
        )
    if template_angles.numel() > 64:
        raise ValueError(
            f"Found {template_angles.numel()} unique orientations. This decoder "
            "expects a small set of discrete templates; bin continuous "
            "orientations first."
        )
    distances = torch.full(
        (template_angles.numel(), n_trials, n_times),
        torch.nan,
        dtype=data.dtype,
        device=data.device,
    )
    folds = _split_indices_by_label(labels.cpu(), n_folds, generator) #每个fold返回不同template想等的trial

    all_indices = torch.arange(n_trials)
    for test_cpu in folds:
        train_mask = torch.ones(n_trials, dtype=torch.bool)
        train_mask[test_cpu] = False
        train_cpu = all_indices[train_mask]
        train = train_cpu.to(data.device)
        test = test_cpu.to(data.device)
        train_labels = labels[train]

        counts = torch.bincount(train_labels, minlength=template_angles.numel())
        sample_count = int(counts.min().item())
        templates = []
        for template_index in range(template_angles.numel()):
            members = train[train_labels == template_index]
            order = torch.randperm(members.numel(), generator=generator)[
                :sample_count
            ].to(data.device)
            templates.append(data[members[order]].mean(dim=0)) #（ntrial，feature，time) -> (feature, time)
        templates = torch.stack(templates) #（template, feature, time)

        train_by_time = data[train].permute(2, 0, 1) #(time, template, feature)
        covariance = _covdiag_batched(train_by_time)
        distances[:, test, :] = _mahalanobis_batched(
            templates, data[test], covariance
        ) #（templates, n_trial, time)

    theta_distance = torch.remainder(theta[:, None]- template_angles[None, :],2 * torch.pi)
    trial_cos = -(torch.cos(theta_distance).T[:, :, None] * distances).mean(dim=0)

    reordered = torch.full_like(distances,torch.nan)

    for template_index in range(template_angles.numel()):
        relative = torch.remainder(
            template_angles - template_angles[template_index],2 * torch.pi)

        order = torch.argsort(relative)
        member_trials = labels == template_index
        reordered[:, member_trials, :] = distances[order][:, member_trials, :]
    distances = -(reordered - reordered.mean(dim=0, keepdim=True))
    return trial_cos, distances


def mahal_decoded(
    features: np.ndarray,
    theta: np.ndarray,
    config: DecodeConfig,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 1,
    return_trialwise: bool = True,
):
    """Run repetitions and optionally return repetition-averaged trial scores."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch cannot access a CUDA device")
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    target = torch.device(device)
    data_t = torch.as_tensor(features, dtype=torch_dtype, device=target) #（trial*feature*time）
    theta_t = torch.as_tensor(theta, dtype=torch_dtype, device=target).flatten()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    trial_cos_sum = torch.zeros(
        (data_t.shape[0], data_t.shape[2]), dtype=torch_dtype, device=target
    ) #（n_trial, n_time)
    distances_sum = torch.zeros(
        (torch.unique(theta_t).numel(), data_t.shape[2]),
        dtype=torch_dtype,
        device=target,
    ) #(n_template, n_time)
    for rep in range(config.reps):
        trial_cos, trial_distances = mahal_theta_kfold(
            data_t, theta_t, config.n_folds, generator
        )
        trial_cos_sum += trial_cos #(ntrial, time)
        distances_sum += trial_distances.mean(dim=1)  #（templates, n_trial, time) -> (templates, time)
        print(f"  repetition {rep + 1}/{config.reps}", flush=True)

    trialwise = trial_cos_sum / config.reps
    cosine = trialwise.mean(dim=0)
    distances = distances_sum / config.reps
    trialwise_np = trialwise.cpu().numpy()
    if config.smooth_ms > 0:
        from scipy.ndimage import gaussian_filter1d

        cosine_np = gaussian_filter1d(
            cosine.cpu().numpy(), config.smooth_ms / config.step_ms
        )
        trialwise_np = gaussian_filter1d(
            trialwise_np,
            config.smooth_ms / config.step_ms,
            axis=1,
        )
    else:
        cosine_np = cosine.cpu().numpy()
    if return_trialwise:
        return cosine_np, distances.cpu().numpy(), trialwise_np
    return cosine_np, distances.cpu().numpy()
