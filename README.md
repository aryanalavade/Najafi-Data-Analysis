# Najafi Lab — Data Analysis

Code for analyzing neural and behavioral data in the study of **predictive processing** in the brain.

---

## Repository Structure

```
Najafi-Data-Analysis/
├── 2p_neural_pipeline/         # Two-photon calcium imaging analysis pipeline
├── SA09_behavioral_analysis/   # Behavioral & DLC analysis for SA09 joystick experiments
├── DLC_SA09_Model_Training/    # DeepLabCut model training for body tracking
└── allen_institute/            # Intro to electrophysiology NWB files (Allen Institute data)
```

---

## Folders

### `2p_neural_pipeline/`
Full pipeline for processing two-photon calcium imaging data. Run in the following order:

| Script | Description |
|--------|-------------|
| `main.py` | Entry point — orchestrates the full pipeline and generates an HTML report |
| `Trialization.py` | Segments raw voltage recordings and dF/F traces into trials aligned to stimulus labels |
| `Alignment.py` | Aligns neural population responses to stimulus events across sessions |
| `StatTest.py` | Runs significance tests to identify stimulus-responsive ROIs |
| `ReadResults.py` | Utility module for reading all data formats (dF/F, voltages, masks, bpod, camera) |
| `visualization1_FieldOfView.py` | Plots surgery window images, ROI masks, and example traces |
| `visualization2_3331Random.py` | Plots results for 3331Random sessions (interval distributions, cluster traces, latent dynamics) |
| `visualization3_1451ShortLong.py` | Plots results for 1451ShortLong sessions |
| `visualization4_4131FixJitterOdd.py` | Plots results for 4131FixJitterOdd sessions |
| `visualization5_3331RandomExtended.py` | Plots results for 3331RandomExtended sessions |

### `SA09_behavioral_analysis/`
Scripts for analyzing behavioral and pupil data from joystick experiments (subject SA09).

| Script | Description |
|--------|-------------|
| `plot_preliminary_figure.py` | Plots pupil position, trajectory, and pupil area aligned to behavioral events, comparing short vs long IPI trials |
| `make_proposal_figures.py` | Same analysis using pre-processed pupil data (Sana's CSV format) — used for grant proposals |
| `print_bpod_events.py` | Utility script to inspect event key names inside a bpod `.mat` session file |

### `DLC_SA09_Model_Training/`
DeepLabCut project files for training a full-body tracking model on subject SA09.

### `allen_institute/`
Introductory notebook for working with electrophysiology data in NWB format using the Allen Institute dataset.

---

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy matplotlib pandas h5py tqdm deeplabcut
```

| Package | Used for |
|---------|----------|
| `numpy` | Array operations throughout |
| `scipy` | Signal filtering, interpolation, stats tests, loading `.mat` files |
| `matplotlib` | All plotting |
| `pandas` | Behavioral data and DLC CSV loading |
| `h5py` | Reading/writing `.h5` data files |
| `tqdm` | Progress bars |
| `deeplabcut` | Video analysis and body tracking (DLC folder) |

---

## Data Formats

- **`ops.npy`** — Suite2p output, one per session
- **`raw_voltages.h5`** — Raw voltage recordings (stimulus, imaging trigger, camera, etc.)
- **`dff.h5`** — dF/F fluorescence traces
- **`neural_trials.h5`** — Trial-segmented neural data (output of `Trialization.py`)
- **`significance.h5`** — ROI significance labels (output of `StatTest.py`)
- **`masks.h5`** — ROI masks and mean/max projection images
- **`bpod_session_data.mat`** — Bpod behavioral session data
- **`camera_*.h5`** — DLC pupil tracking results

---

## Contact

For questions about this code, contact the Najafi Lab.
