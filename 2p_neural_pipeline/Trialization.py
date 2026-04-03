#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.interpolate import interp1d

from mymodules.ReadResults import read_raw_voltages
from mymodules.ReadResults import read_dff
from mymodules.ReadResults import read_camera
from mymodules.ReadResults import read_bpod_mat_data

# remove trial start trigger voltage impulse.
def remove_start_impulse(vol_time, vol_stim_vis):
    min_duration = 100
    changes = np.diff(vol_stim_vis.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1
    if vol_stim_vis[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if vol_stim_vis[-1] == 1:
        end_indices = np.append(end_indices, len(vol_stim_vis))
    for start, end in zip(start_indices, end_indices):
        duration = vol_time[end-1] - vol_time[start]
        if duration < min_duration:
            vol_stim_vis[start:end] = 0
    return vol_stim_vis

# correct beginning vol_stim_vis if not start from 0.
def correct_vol_start(vol_stim_vis):
    if vol_stim_vis[0] == 1:
        vol_stim_vis[:np.where(vol_stim_vis==0)[0][0]] = 0
    return vol_stim_vis

# detect the rising edge and falling edge of binary series.
def get_trigger_time(
        vol_time,
        vol_bin
        ):
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # select the indice for risging and falling.
    # give the edges in ms.
    time_up   = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down

# correct the fluorescence signal timing.
def correct_time_img_center(time_img):
    # find the frame internal.
    diff_time_img = np.diff(time_img, append=0)
    # correct the last element.
    diff_time_img[-1] = np.mean(diff_time_img[:-1])
    # move the image timing to the center of photon integration interval.
    diff_time_img = diff_time_img / 2
    # correct each individual timing.
    time_neuro = time_img + diff_time_img
    return time_neuro

# get stimulus sequence labels.
def get_stim_labels(bpod_sess_data, vol_time, vol_stim_vis):
    stim_time_up, stim_time_down = get_trigger_time(vol_time, vol_stim_vis)
    if bpod_sess_data['img_seq_label'][-1] == -1:
        stim_time_up = stim_time_up[:-1]
        stim_time_down = stim_time_down[:-1]
    stim_labels = np.zeros((len(stim_time_up), 8))
    # row 0: stim start.
    # row 1: stim end.
    # row 2: img_seq_label.
    # row 3: standard_types.
    # row 4: fix_jitter_types.
    # row 5: oddball_types.
    # row 6: random_types.
    # row 7: opto_types.
    stim_labels[:,0] = stim_time_up
    stim_labels[:,1] = stim_time_down
    stim_labels[:,2] = bpod_sess_data['img_seq_label']
    stim_labels[:,3] = bpod_sess_data['standard_types']
    stim_labels[:,4] = bpod_sess_data['fix_jitter_types']
    stim_labels[:,5] = bpod_sess_data['oddball_types']
    stim_labels[:,6] = bpod_sess_data['random_types']
    stim_labels[:,7] = bpod_sess_data['opto_types']
    return stim_labels
    
# save trial neural data.
def save_trials(
        ops, time_neuro, dff, stim_labels,
        vol_time, vol_start, vol_stim_vis,
        vol_stim_aud, vol_flir,
        vol_pmt, vol_led,
        camera_time, camera_pupil
        ):
    # file structure:
    # ops['save_path0'] / neural_trials.h5
    # ---- time
    # ---- stim
    # ---- dff
    # ---- vol_stim
    # ---- vol_time
    # ---- stim_labels
    # ...
    h5_path = os.path.join(ops['save_path0'], 'neural_trials.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)
    f = h5py.File(h5_path, 'w')
    grp = f.create_group('neural_trials')
    grp['time']         = time_neuro
    grp['dff']          = dff
    grp['stim_labels']  = stim_labels
    grp['vol_time']     = vol_time
    grp['vol_start']     = vol_start
    grp['vol_stim_vis'] = vol_stim_vis
    grp['vol_stim_aud'] = vol_stim_aud
    grp['vol_flir']     = vol_flir
    grp['vol_pmt']      = vol_pmt
    grp['vol_led']      = vol_led
    grp['camera_time']  = camera_time
    grp['camera_pupil'] = camera_pupil
    f.close()

# main function for trialization.
def run(ops):
    print('Reading dff traces and voltage recordings')
    dff = read_dff(ops)
    [vol_time, vol_start, vol_stim_vis, vol_img,
     vol_hifi, vol_stim_aud, vol_flir,
     vol_pmt, vol_led] = read_raw_voltages(ops)
    vol_stim_vis = remove_start_impulse(vol_time, vol_stim_vis)
    vol_stim_vis = correct_vol_start(vol_stim_vis)
    bpod_sess_data = read_bpod_mat_data(ops)
    print('Correcting 2p camera trigger time')
    # signal trigger time stamps.
    time_img, _   = get_trigger_time(vol_time, vol_img)
    # correct imaging timing.
    time_neuro = correct_time_img_center(time_img)
    # stimulus sequence labeling.
    stim_labels = get_stim_labels(bpod_sess_data, vol_time, vol_stim_vis)
    # processing dlc results.
    camera_time, camera_pupil = read_camera(ops)
    camera_time = correct_time_img_center(camera_time)
    camera_pupil = interp1d(camera_time, camera_pupil, bounds_error=False, fill_value=np.nan)(time_neuro)
    # save the final data.
    print('Saving trial data')
    save_trials(
        ops, time_neuro, dff, stim_labels,
        vol_time, vol_start, vol_stim_vis,
        vol_stim_aud, vol_flir,
        vol_pmt, vol_led,
        camera_time, camera_pupil)


# %%

def plot_vol_img_segment(
    vol_time: np.ndarray,
    vol_img: np.ndarray,
    t_start: float | None = None,
    t_end: float | None = None,
    *,
    idx_start: int | None = None,
    idx_end: int | None = None,
    show_edges: bool = True,
    downsample: int | None = None,
    figsize: tuple[int, int] = (10, 2.2),
    color: str = "C0",
    edge_color: str = "r",
    lw: float = 0.8,
    show: bool = True,
    savepath: str | None = None,
    title: str | None = None,
):
    """
    Plot a segment of the imaging trigger (vol_img) against vol_time.

    Selection:
      - Provide t_start / t_end (seconds) to pick nearest times, OR
      - Provide idx_start / idx_end (indices). Time args take priority if given.

    Parameters
    ----------
    vol_time : 1D array of timestamps (seconds).
    vol_img  : 1D array (same length) binary/analog imaging trigger.
    downsample : int stride to thin the plotted points (None keeps all).
    show_edges : mark rising/falling edges (detected on threshold 0.5).
    """
    import matplotlib.pyplot as plt

    if vol_time.shape != vol_img.shape:
        raise ValueError("vol_time and vol_img must have identical shape.")

    n = len(vol_time)
    if t_start is not None or t_end is not None:
        # nearest index helper
        def _nearest_idx(tvec, t):
            if t <= tvec[0]: return 0
            if t >= tvec[-1]: return len(tvec) - 1
            i = int(np.searchsorted(tvec, t))
            if i == 0: return 0
            if i >= len(tvec): return len(tvec) - 1
            return i - 1 if abs(tvec[i - 1] - t) <= abs(tvec[i] - t) else i
        i0 = 0 if t_start is None else _nearest_idx(vol_time, float(t_start))
        i1 = n - 1 if t_end is None else _nearest_idx(vol_time, float(t_end))
    else:
        i0 = 0 if idx_start is None else int(idx_start)
        i1 = n - 1 if idx_end is None else int(idx_end)

    if i0 < 0 or i1 >= n:
        raise IndexError("Index window out of bounds.")
    if i1 < i0:
        i0, i1 = i1, i0

    t_seg = vol_time[i0:i1 + 1]
    v_seg = vol_img[i0:i1 + 1]

    if downsample and downsample > 1:
        t_seg = t_seg[::downsample]
        v_seg = v_seg[::downsample]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if title:
        ax.set_title(title)

    ax.plot(t_seg, v_seg, color=color, lw=lw)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("vol_img")

    if show_edges:
        # detect edges on the full (un-downsampled) segment for accuracy
        raw = vol_img[i0:i1 + 1]
        diff = np.diff(raw.astype(int), prepend=0)
        idx_up = np.where(diff == 1)[0]
        idx_dn = np.where(diff == -1)[0]
        for iu in idx_up:
            ax.axvline(vol_time[i0 + iu], color=edge_color, ls='--', lw=0.6, alpha=0.8)
        for idn in idx_dn:
            ax.axvline(vol_time[i0 + idn], color=edge_color, ls=':', lw=0.6, alpha=0.6)

    ax.set_xlim(t_seg[0], t_seg[-1])
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()
    return fig, ax

# Example:
# fig, ax = plot_vol_img_segment(vol_time, vol_img, t_start=0.0, t_end=1000.0, show_edges=True)
# fig, ax = plot_vol_img_segment(vol_time, vol_img, idx_start=200000, idx_end=210000, downsample=5)

