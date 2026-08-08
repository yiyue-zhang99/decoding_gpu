
import numpy as np
import finufft


# ── 核心批处理函数 ──────────────────────────────────────────────────────────────

def _nufft2d_batch(data, sampling_rate, chn_dist,
                   freq_lo=1.0, freq_hi=40.0):
    """
    Parameters
    ----------
    data          : ndarray, shape (n_elec, n_times, n_batch)
    sampling_rate : float [Hz]
    chn_dist      : array-like, shape (n_elec,)  非均匀空间坐标（mm 或弧长）
    freq_lo, freq_hi : float  保留的时间频率范围 [Hz]
    hann_taper    : bool  是否沿时间轴加 Hann 窗

    Returns
    -------
    fw : (n_freq, n_batch)   前向波（负空间频率）的最大谱值
    bw : (n_freq, n_batch)   后向波（正空间频率）的最大谱值
    ff : (n_freq,)           对应时间频率轴 [Hz]
    """
    n_elec, n_times, n_batch = data.shape
    chn_dist = np.asarray(chn_dist, dtype=np.float64)


    # 空间坐标归一化到 [-π, π)
    s_min, s_max = chn_dist.min(), chn_dist.max()
    s_norm = (-np.pi + (chn_dist - s_min) / (s_max - s_min) * 2 * np.pi
                  ).astype(np.float64)


    # MATLAB:
    # x = linspace(-pi, pi, n)
    t_norm = np.linspace(-np.pi, np.pi, n_times, endpoint=False, dtype=np.float64)

    # MATLAB:
    # [x_grid, y_grid] = meshgrid(x, y_normalized);
    # finufft2d1(y_grid(:), x_grid(:), data(:), ...)
    # (elec1,t1), (elec2,t1), ..., (elecN,t1), (elec1,t2), ...

 

    xt, xs = np.meshgrid(t_norm, s_norm)
    xs = np.ascontiguousarray(xs.ravel(order="F"))
    xt = np.ascontiguousarray(xt.ravel(order="F"))

    # 强度矩阵 c: 与 MATLAB data(:) 
    c = np.asfortranarray(data).reshape(n_elec * n_times, n_batch, order="F").T
    c = np.ascontiguousarray(c.astype(np.complex128))

    # 2D NUFFT type-1 → fhat: (n_batch, n_elec, n_times)
    # fhat: (n_batch, n_elec, n_times)
    # xs: spatial data, n_elec
    # xt: time data, n_times
    # isign=+1 
    fhat = finufft.nufft2d1(xs, xt, c,
                             n_modes=(n_elec, n_times),
                             isign=+1,
                             eps=1e-9)   # (n_batch, n_elec, n_times)
    fhat = np.abs(fhat)   # (n_batch, n_elec, n_times)

    fx = np.fft.fftshift(np.fft.fftfreq(n_times, d=1.0 / sampling_rate))
    fy = np.fft.fftshift(np.fft.fftfreq(n_elec)) * n_elec

    freq_mask = (fx >= freq_lo) & (fx <= freq_hi)
    ff = fx[freq_mask]

    fhat_sel = fhat[:, :, freq_mask]

    # 前向波: 负空间频率方向取最大；后向波: 正空间频率方向取最大
    fw = fhat_sel[:, fy < 0, :].max(axis=1).T   # (n_freq, n_batch)
    bw = fhat_sel[:, fy > 0, :].max(axis=1).T   # (n_freq, n_batch)

    return fw, bw, ff




# ── 滑动窗口完整分析 ──────────────────────────────────────────────────────────

def wave_trigger_computer_nonuniform(the_data,
                              electrode_lines,
                              chn_dist_lines,
                              sfreq=500,
                              window_size=250,
                              step=50,
                              shuffle_reps=10,
                              freq_lo=1.0,
                              freq_hi=40.0,
                              demean=False):
    """

    Parameters
    ----------
    the_data        : (n_channels, n_times, n_trials)
    electrode_lines : list of lists  — channel indices per line (0-based)
    chn_dist_lines  : list of array-like — physical positions along each line
    sfreq           : float [Hz]
    window_size     : int  滑动窗口长度（采样点数）
    step            : int  窗口步长（采样点数）
    shuffle_reps    : int  surrogate 置换次数
    freq_lo, freq_hi : float  保留频段 [Hz]
    demean         : bool   是否对每个窗口的数据进行空间和时间双重去均值

    Returns
    -------
    fwMax   : (n_lines, n_freqs, n_win, n_trials)  前向波最大谱值
    bwMax   : (n_lines, n_freqs, n_win, n_trials)  后向波最大谱值
    fwMaxSS : (n_lines, n_freqs, n_win, n_trials)  前向波 surrogate 均值
    bwMaxSS : (n_lines, n_freqs, n_win, n_trials)  后向波 surrogate 均值
    starts  : (n_win,)                              各窗口起始采样点
    ff      : (n_freqs,)                            频率轴 [Hz]
    """
    n_ch, n_times, n_trials = the_data.shape
    starts  = list(range(0, n_times - window_size + 1, step))
    n_lines = len(electrode_lines)
    n_win   = len(starts)

    # 预先确定 n_freqs
    idx0 = np.asarray(electrode_lines[0], dtype=int)
    _, _, ff = _nufft2d_batch(
        the_data[idx0, :window_size, :1], sfreq, chn_dist_lines[0],
        freq_lo=freq_lo, freq_hi=freq_hi
    )
    n_freqs = len(ff)

    fwMax   = np.full((n_lines, n_freqs, n_win, n_trials), np.nan)
    bwMax   = np.full((n_lines, n_freqs, n_win, n_trials), np.nan)
    fwMaxSS = np.full((n_lines, n_freqs, n_win, n_trials), np.nan)
    bwMaxSS = np.full((n_lines, n_freqs, n_win, n_trials), np.nan)

    for wi, st in enumerate(starts):
        seg = slice(st, st + window_size)
        for mm in range(n_lines):
            idx_line  = np.asarray(electrode_lines[mm], dtype=int)
            dist_line = np.asarray(chn_dist_lines[mm],  dtype=float)
            n_elec    = len(idx_line)
            seg_data  = the_data[idx_line][:, seg, :]  # (n_elec, window_size, n_trials)
            # spatial demean and temporal demean 
            if demean:
                seg_data = seg_data - seg_data.mean(axis=0, keepdims=True)
                # seg_data = seg_data - seg_data.mean(axis=1, keepdims=True)

            # ── 真实波：batch over n_trials ───────────────────────────────────
            fw, bw, _ = _nufft2d_batch(seg_data, sfreq, dist_line,
                                        freq_lo=freq_lo, freq_hi=freq_hi)
            fwMax[mm, :, wi, :] = fw   # (n_freqs, n_trials)
            bwMax[mm, :, wi, :] = bw

            # ── surrogate：每个 trial、每次迭代都独立置换，严格对齐 MATLAB ─────
            surr_data = np.empty(
                (n_elec, window_size, shuffle_reps * n_trials),
                dtype=seg_data.dtype
            )
            for tt in range(n_trials):
                for kk in range(shuffle_reps):
                    perm = np.random.permutation(n_elec)
                    out_idx = kk * n_trials + tt
                    surr_data[:, :, out_idx] = seg_data[perm, :, tt]

            fw_s, bw_s, _ = _nufft2d_batch(surr_data, sfreq, dist_line,
                                             freq_lo=freq_lo, freq_hi=freq_hi)
            # fw_s: (n_freqs, shuffle_reps * n_trials)
            fw_s = fw_s.reshape(n_freqs, shuffle_reps, n_trials)
            bw_s = bw_s.reshape(n_freqs, shuffle_reps, n_trials)
            fwMaxSS[mm, :, wi, :] = fw_s.mean(axis=1)
            bwMaxSS[mm, :, wi, :] = bw_s.mean(axis=1)

    return fwMax, bwMax, fwMaxSS, bwMaxSS, np.array(starts), ff


# ── 电极距离工具函数 ──────────────────────────────────────────────────────────

def elec_distance(pos, xo=0.0, yo=0.0, zo=0.0):
    """
    Parameters
    ----------
    pos : array-like, shape (n_elec, 3)
        每个电极的 (X, Y, Z) 笛卡尔坐标。
    xo, yo, zo : float
        球心坐标，默认 (0, 0, 0)。

    Returns
    -------
    dist : ndarray, shape (n_elec, n_elec)
        两两电极之间的球面弧长（与坐标单位相同）。
    """
    pos = np.asarray(pos, dtype=float)
    A = pos - np.array([xo, yo, zo])
    lengths = np.linalg.norm(A, axis=1)

    n = len(lengths)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r     = (lengths[i] + lengths[j]) / 2.0
            denom = lengths[i] * lengths[j]
            if denom == 0.0:
                dist[i, j] = np.linalg.norm(A[i] - A[j])
                continue
            cos_t = np.dot(A[i], A[j]) / denom
            cos_t = np.clip(cos_t, -1.0, 1.0)
            theta = np.arccos(cos_t)
            dist[i, j] = r * theta
    return dist


def build_chn_dist_lines(electrode_lines, pos, xo=0.0, yo=0.0, zo=0.0):
    """
    Parameters
    ----------
    electrode_lines : list of lists
        每条线的电极 channel index（0-based）。
    pos : array-like, shape (n_total_channels, 3)
        所有通道的 (X, Y, Z) 坐标，顺序与 data 的 channel 轴对应。

    Returns
    -------
    chn_dist_lines : list of ndarray
        每条线各电极的逐段累积弧长（与 MATLAB compute_waves 一致）。
    """
    pos = np.asarray(pos, dtype=float)
    dist_matrix = elec_distance(pos, xo, yo, zo)

    chn_dist_lines = []
    for line in electrode_lines:
        idx = np.asarray(line, dtype=int)
        d = np.zeros(len(idx))
        for k in range(1, len(idx)):
            d[k] = d[k-1] + dist_matrix[idx[k-1], idx[k]]
        chn_dist_lines.append(d)
    return chn_dist_lines
