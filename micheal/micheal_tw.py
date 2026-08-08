# %%
import mne
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/dilay/project2/tw/travelling_waves/tw/tw_fft/func')
from wave_ana_tw_power2 import *
import os
from scipy.ndimage import gaussian_filter
from mne.stats import permutation_cluster_1samp_test
import pickle
from joblib import Parallel, delayed
import glob, os, re
from collections import defaultdict



def peel_to_str(x):
    while isinstance(x, np.ndarray):
        x = x.item() if x.size == 1 else x.ravel()[0]
    return str(x)


def build_electrode_lines(ch_names):
    M  = ['Oz', 'POz', 'Pz', 'CPz', 'Cz', 'FCz', 'Fz']

    L1 = ['O1', 'PO3', 'P1', 'CP1', 'C1', 'FCz', 'Fz']
    L2 = ['O1', 'PO3', 'P1', 'CP1', 'C1', 'FC1', 'F1']
    L3 = ['PO7','P5','CP3','C3','FC1','F1']         
    L4 = ['O1','PO3','P3','CP3','C3','FC3','F3']
    L5 = ['PO7','P5','CP3','C3','FC3','F3']

    R1 = ['O2', 'PO4', 'P2', 'CP2', 'C2', 'FCz', 'Fz']
    R2 = ['O2', 'PO4', 'P2', 'CP2', 'C2', 'FC2', 'F2']
    R3 = ['PO8','P6','CP4','C4','FC2','F2']                                
    R4 = ['O2','PO4','P4','CP4','C4','FC4','F4']
    R5 = ['PO8','P6','CP4','C4','FC4','F4']

    lines = [L5, L4, L3, L2, L1, M, R1, R2, R3, R4, R5]
    names = ['L5','L4','L3','L2','L1','M','R1','R2','R3','R4','R5']


    name2idx = {c: i for i, c in enumerate(ch_names)}

    electrode_lines = []
    for nm, line in zip(names, lines):
        missing = [c for c in line if c not in name2idx]
        if missing:
            print(f'[build_electrode_lines] {nm} missing: {missing} -> dropped')
        idxs = [name2idx[c] for c in line if c in name2idx]
        electrode_lines.append(idxs)
    return electrode_lines, names


def load_and_compute_one(mat_path,
                         sfreq=500, window_size=250, step=25, shuffle_reps=200):
    eeg = loadmat(mat_path, struct_as_record=False)
    data = eeg['ft_mem'].item()

    # ── 时间裁剪 ─────────────────────────────────────────────────────────────
    t = np.asarray(data.time).ravel()
    tmin, tmax = -1.2, 6.0
    mask = (t >= tmin) & (t <= tmax)
    eeg_crop = data.trial[:, :, mask]   # (n_trials, n_ch, n_times_crop)
    eeg_data = eeg_crop #- eeg_crop.mean(axis=2, keepdims=True)
    
    chanel   = data.elec.item()
    ch_names = [peel_to_str(x) for x in np.asarray(chanel.label).ravel()]
    elecpos = np.asarray(chanel.elecpos)   # (n_ch, 3)

    # ── bad trials ────────────────────────────────────────────────────────────
    # cue_loc    = data.Results[0, 2]
    bad_trials = np.unique(np.concatenate([
        data.bad_trials_mem.reshape(-1),
        data.bad_trials_imp1.reshape(-1),
        data.bad_trials_imp2.reshape(-1),
        data.bad_trials_probe1.reshape(-1),
        data.bad_trials_probe2.reshape(-1)
    ]))
    eeg_data = np.delete(eeg_data, bad_trials - 1, axis=0)
    early_rot = np.where(np.asarray(data.Results[:, 8]).ravel() > 0, 2, 1)
    late_rot  = np.where(np.asarray(data.Results[:, 12]).ravel() > 0, 2, 1)
    early_rot = np.delete(early_rot, bad_trials - 1)
    late_rot  = np.delete(late_rot,  bad_trials - 1)

    early_acc =  np.asarray(data.Results[:, 10]).ravel() 
    early_rt = np.asarray(data.Results[:, 9]).ravel() 
    late_acc =  np.asarray(data.Results[:, 14]).ravel() 
    late_rt = np.asarray(data.Results[:, 13]).ravel() 
    early_acc = np.delete(early_acc, bad_trials-1)
    early_rt = np.delete(early_rt, bad_trials-1)
    late_acc = np.delete(late_acc, bad_trials-1)
    late_rt = np.delete(late_rt, bad_trials -1)


    cue_loc = np.asarray(data.Results[:, 2]).ravel()
    cue_loc = np.delete(cue_loc, bad_trials-1)

    # ── 电极线 & 距离 ─────────────────────────────────────────────────────────
    electrode_lines, _ = build_electrode_lines(ch_names)

    # 用 elecpos 计算每条线上的真实球面弧长
    # chn_dist_lines = build_chn_dist_lines(electrode_lines, elecpos)

    # ── traveling wave 计算（dense 2D NUFFT Type-3）──────────────────────────
    the_data = eeg_data.transpose(1, 2, 0)   # (n_ch, n_times, n_trials)

    fwmax, bwmax, fwssmax, bwssmax, fftf, time, ff = wave_trigger_computer_uniform(
        the_data,
        electrode_lines,
        sfreq=sfreq,
        window_size=window_size,
        step=step,
        shuffle_reps=shuffle_reps,
        baseline_mode='surr',
        spatial_demean=False,
        max_batch=10000
    )



    return fwmax, fwssmax, bwmax, bwssmax, fftf, time, ff, cue_loc, early_rot,late_rot, early_acc, early_rt, late_acc,late_rt


# %%
root      = '/home/dilay/project2/tw/Data/Micheal_Data_exp2/'
mat_files = sorted(glob.glob(os.path.join(root, 'MemImp3_mem_whole_sess*_*.mat')))

CACHE_DIR = '/home/dilay/project2/tw/results/micheal_fft/'
os.makedirs(CACHE_DIR, exist_ok=True)

def run_one_file(f):
    base = os.path.basename(f)
    sess = int(re.search(r'sess(\d+)', base).group(1))
    subj = int(re.search(r'_(\d+)\.mat$', base).group(1))
    out_path = os.path.join(CACHE_DIR, f'subj{subj:02d}_sess{sess}.pkl')

    if os.path.exists(out_path):
        print(f'[skip] {base}')
        return subj, sess, out_path

    fwmax, fwssmax, bwmax, bwssmax, fftf, time, ff, cue_loc, \
        early_rot, late_rot, early_acc, early_rt, late_acc, late_rt = load_and_compute_one(f)

    result = dict(
        fwmax=fwmax, fwssmax=fwssmax, bwmax=bwmax, bwssmax=bwssmax,
        fftf=fftf, time=time, ff=ff, cue_loc=cue_loc,
        early_rot=early_rot, late_rot=late_rot,
        early_acc=early_acc, early_rt=early_rt,
        late_acc=late_acc, late_rt=late_rt,
    )
    with open(out_path, 'wb') as fp:
        pickle.dump(result, fp)
    print(f'[done] {base}')
    return subj, sess, out_path

file_index = Parallel(n_jobs=12, verbose=10)(
    delayed(run_one_file)(f) for f in mat_files
)


