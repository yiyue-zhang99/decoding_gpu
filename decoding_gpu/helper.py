"""Shared feature-building and Mahalanobis helpers for PyTorch decoders."""

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

    # Timepoint decoding means exactly one feature vector per original sample.
    # It must not inherit the temporal stride used by sliding-window decoding.
    if mode == "timepoint":
        return np.asarray(data[:, :, eligible]), time[eligible]

    requested = np.arange(
        time[eligible[0]],
        time[eligible[-1]] + config.step_ms / 2000,
        config.step_ms / 1000,
    ) #每次间隔step_ms取点1
    indices = np.array([np.argmin(np.abs(time - value)) for value in requested]) #找出切片点所在的
    indices = np.unique(indices) #decoding过后保留的时间点

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
