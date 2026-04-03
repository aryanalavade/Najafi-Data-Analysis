import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  # <-- Back to using sio
import h5py
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D 

# --- 1. CONFIGURATION ---

# This ONE file from Sana has all the pupil/eye data
SANA_PROCESSED_FILE = 'SA01_20250430_Joystick_VG_area.csv' 

# --- !! MODIFIED !! ---
# We are NOW using the .mat file for events
EVENTS_FILE_PATH = 'bpod_session_data.mat' 
# --- !! END MODIFIED !! ---

BODYPART_TO_PLOT = 'PupilCenter' 
COORDINATES_TO_PLOT = ['x', 'y'] 

VIDEO_FPS = 30  # Assumes timestamps in .mat file are in seconds
TIME_WINDOW_SECONDS = 2
TIME_WINDOW_FRAMES = int(TIME_WINDOW_SECONDS * VIDEO_FPS / 2)
LIKELIHOOD_THRESHOLD = 0.9 
SMOOTHING_WINDOW = 5 
TRIAL_TYPE_SHORT = 1  # Assuming 1=Short (from your original script)

# --- 2. DEFINE PLOTTING FUNCTIONS ---
# --- (UNCHANGED) ---

def plot_aligned_data(kinematic_data, event_frames, is_short_trial, trial_types_found, bodypart, coordinate, event_name):
    """Plots kinematic data aligned to an event (Time Series Plot)."""
    
    kinematic_data[dlc_data[bodypart]['likelihood'] < LIKELIHOOD_THRESHOLD] = np.nan
    
    short_trial_kinematics = []
    long_trial_kinematics = []

    for frame_index, is_short in zip(event_frames, is_short_trial):
        start_frame = frame_index - TIME_WINDOW_FRAMES
        end_frame = frame_index + TIME_WINDOW_FRAMES
        
        # This check is the one that was failing. Let's see if it works now.
        if start_frame > 0 and end_frame < len(kinematic_data):
            data_snippet = kinematic_data.iloc[start_frame:end_frame].values
            if is_short:
                short_trial_kinematics.append(data_snippet)
            else:
                long_trial_kinematics.append(data_snippet)

    print(f"  Found {len(short_trial_kinematics)} short trials and {len(long_trial_kinematics)} long trials for this event.")

    if len(short_trial_kinematics) > 0:
        mean_short = np.nanmean(short_trial_kinematics, axis=0)
        sem_short = np.nanstd(short_trial_kinematics, axis=0) / np.sqrt(len(short_trial_kinematics))
    else:
        mean_short = np.full(TIME_WINDOW_FRAMES * 2, np.nan)
        sem_short = np.full(TIME_WINDOW_FRAMES * 2, np.nan)

    if len(long_trial_kinematics) > 0:
        mean_long = np.nanmean(long_trial_kinematics, axis=0)
        sem_long = np.nanstd(long_trial_kinematics, axis=0) / np.sqrt(len(long_trial_kinematics))
    else:
        mean_long = np.full(TIME_WINDOW_FRAMES * 2, np.nan)
        sem_long = np.full(TIME_WINDOW_FRAMES * 2, np.nan)

    time_axis = np.linspace(-TIME_WINDOW_SECONDS/2, TIME_WINDOW_SECONDS/2, len(mean_short))

    if len(short_trial_kinematics) > 0: mean_short = pd.Series(mean_short).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values
    if len(long_trial_kinematics) > 0: mean_long = pd.Series(mean_long).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values

    plt.figure(figsize=(10, 6))
    plt.title(f"Preliminary Data: {bodypart} '{coordinate}' Center (Aligned to {event_name})", fontsize=16)
    plt.xlabel(f"Time from {event_name} (seconds)", fontsize=12)
    plt.ylabel(f"{bodypart} '{coordinate}' Center (pixels)", fontsize=12)

    plt.plot(time_axis, mean_long, color='red', label=f'Long IPI Trials (n={len(long_trial_kinematics)})')
    plt.fill_between(time_axis, mean_long - sem_long, mean_long + sem_long, color='red', alpha=0.2)
    plt.plot(time_axis, mean_short, color='blue', label=f'Short IPI Trials (n={len(short_trial_kinematics)})')
    plt.fill_between(time_axis, mean_short - sem_short, mean_short + sem_short, color='blue', alpha=0.2)
    plt.axvline(0, color='black', linestyle='--') 
    
    start_index, event_index = 0, TIME_WINDOW_FRAMES
    if not np.all(np.isnan(mean_long)):
        plt.plot(time_axis[start_index], mean_long[start_index], 'o', c='black', mfc='red', ms=10) 
        plt.plot(time_axis[event_index], mean_long[event_index], 'X', c='black', mfc='red', ms=12) 
    if not np.all(np.isnan(mean_short)):
        plt.plot(time_axis[start_index], mean_short[start_index], 'o', c='black', mfc='blue', ms=10)
        plt.plot(time_axis[event_index], mean_short[event_index], 'X', c='black', mfc='blue', ms=12)

    handles, labels = plt.gca().get_legend_handles_labels() 
    if 'VisStim' in event_name: event_label = 'Stimulus Onset (0.0s)'
    else: event_label = f'{event_name} (0.0s)'
    handles.extend([
        Line2D([0], [0], color='black', linestyle='--'), 
        Line2D([0], [0], marker='o', color='none', mfc='black', mec='black', ms=10), 
        Line2D([0], [0], marker='X', color='none', mfc='black', mec='black', ms=10) 
    ])
    labels.extend([f'Aligned to {event_name} Event', 'Start (-1.0s)', event_label])
    by_label = dict(zip(labels, handles))
    plt.legend(handles=by_label.values(), labels=by_label.keys())
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = f'my_preliminary_figure_{coordinate}_pos_{event_name}.png'
    plt.savefig(output_filename); plt.close()


def plot_trajectory_data(event_frames, is_short_trial, bodypart, event_name):
    """Plots kinematic data as a "Time Trajectory" (X vs Y Plot)."""
    
    x_data, y_data = dlc_data[bodypart]['x'].copy(), dlc_data[bodypart]['y'].copy()
    likelihood = dlc_data[bodypart]['likelihood']
    x_data[likelihood < LIKELIHOOD_THRESHOLD] = np.nan
    y_data[likelihood < LIKELIHOOD_THRESHOLD] = np.nan

    short_x, short_y, long_x, long_y = [], [], [], []

    for frame_index, is_short in zip(event_frames, is_short_trial):
        start_frame, end_frame = frame_index - TIME_WINDOW_FRAMES, frame_index + TIME_WINDOW_FRAMES
        if start_frame > 0 and end_frame < len(x_data):
            if is_short:
                short_x.append(x_data.iloc[start_frame:end_frame].values)
                short_y.append(y_data.iloc[start_frame:end_frame].values)
            else:
                long_x.append(x_data.iloc[start_frame:end_frame].values)
                long_y.append(y_data.iloc[start_frame:end_frame].values)

    if len(short_x) > 0: mean_short_x, mean_short_y = np.nanmean(short_x, axis=0), np.nanmean(short_y, axis=0)
    else: mean_short_x, mean_short_y = [], []
    if len(long_x) > 0: mean_long_x, mean_long_y = np.nanmean(long_x, axis=0), np.nanmean(long_y, axis=0)
    else: mean_long_x, mean_long_y = [], []
    if len(mean_short_x) > 0:
        mean_short_x = pd.Series(mean_short_x).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values
        mean_short_y = pd.Series(mean_short_y).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values
    if len(mean_long_x) > 0:
        mean_long_x = pd.Series(mean_long_x).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values
        mean_long_y = pd.Series(mean_long_y).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values

    if len(mean_short_x) == 0 and len(mean_long_x) == 0:
        print(f"  Skipping Trajectory plot for '{event_name}': No trial data found.")
        return 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    t_len = len(mean_short_x) if len(mean_short_x) > 0 else len(mean_long_x)
    if t_len == 0: t_len = len(mean_long_x) if len(mean_long_x) > 0 else len(mean_short_x)
    t = np.linspace(-TIME_WINDOW_SECONDS/2, TIME_WINDOW_SECONDS/2, t_len)
    norm = plt.Normalize(t.min(), t.max()); cmap = 'viridis' 

    def colorline(ax, x, y, t, cmap, linestyle):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles=linestyle)
        lc.set_array(t); lc.set_linewidth(3) 
        line = ax.add_collection(lc); return line

    if len(mean_short_x) > 0:
        colorline(ax1, mean_short_x, mean_short_y, t, cmap, 'solid')
        ax1.plot(mean_short_x[0], mean_short_y[0], 'o', c='black', mfc='white', ms=10) # Start
        ax1.plot(mean_short_x[TIME_WINDOW_FRAMES], mean_short_y[TIME_WINDOW_FRAMES], 'X', c='black', mfc='white', ms=12) # Event
        ax1.set_title('Short IPI Trajectory', fontsize=14)
    ax1.set_xlabel(f"{bodypart} 'x' Position (pixels)", fontsize=12); ax1.set_ylabel(f"{bodypart} 'y' Position (pixels)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6); ax1.set_aspect('equal', adjustable='box') 

    if len(mean_long_x) > 0:
        colorline(ax2, mean_long_x, mean_long_y, t, cmap, 'solid')
        ax2.plot(mean_long_x[0], mean_long_y[0], 'o', c='black', mfc='white', ms=10) # Start
        ax2.plot(mean_long_x[TIME_WINDOW_FRAMES], mean_long_y[TIME_WINDOW_FRAMES], 'X', c='black', mfc='white', ms=12) # Event
        ax2.set_title('Long IPI Trajectory', fontsize=14)
    ax2.set_xlabel(f"{bodypart} 'x' Position (pixels)", fontsize=12); ax2.set_ylabel(f"{bodypart} 'y' Position (pixels)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6); ax2.set_aspect('equal', adjustable='box') 

    fig.suptitle(f"Preliminary Data: {bodypart} X-Y Trajectory (Aligned to {event_name})", fontsize=18)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax1, ax2], location='right', shrink=0.8)
    cbar.set_label('Time from Event (seconds)')
    if 'VisStim' in event_name: event_label = 'Stimulus Onset (0.0s)'
    else: event_label = f'{event_name} (0.0s)'
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', mfc='black', mec='black', ms=10, label='Start (-1.0s)'),
        Line2D([0], [0], marker='X', color='none', mfc='black', mec='black', ms=10, label=event_label)
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.1, 0.9))
    output_filename = f'my_preliminary_figure_TRAJECTORY_{event_name}.png'
    plt.savefig(output_filename); plt.close()


def plot_pupil_area(event_frames, is_short_trial, event_name):
    """Plots pupil area (arousal) aligned to an event."""
    
    try:
        pupil_area = dlc_data['PupilArea']['area'].copy()
        likelihood = dlc_data['PupilArea']['likelihood']
        pupil_area[likelihood < LIKELIHOOD_THRESHOLD] = np.nan
    except KeyError:
        print(f"  Skipping Pupil Area plot: 'PupilArea' data not found.")
        return
        
    short_trial_kinematics, long_trial_kinematics = [], []

    for frame_index, is_short in zip(event_frames, is_short_trial):
        start_frame, end_frame = frame_index - TIME_WINDOW_FRAMES, frame_index + TIME_WINDOW_FRAMES
        if start_frame > 0 and end_frame < len(pupil_area):
            data_snippet = pupil_area.iloc[start_frame:end_frame].values
            if is_short: short_trial_kinematics.append(data_snippet)
            else: long_trial_kinematics.append(data_snippet)

    print(f"  Found {len(short_trial_kinematics)} short (Area) and {len(long_trial_kinematics)} long (Area) trials.")

    if len(short_trial_kinematics) > 0:
        mean_short = np.nanmean(short_trial_kinematics, axis=0)
        sem_short = np.nanstd(short_trial_kinematics, axis=0) / np.sqrt(len(short_trial_kinematics))
    else:
        mean_short, sem_short = np.full(TIME_WINDOW_FRAMES * 2, np.nan), np.full(TIME_WINDOW_FRAMES * 2, np.nan)
    if len(long_trial_kinematics) > 0:
        mean_long = np.nanmean(long_trial_kinematics, axis=0)
        sem_long = np.nanstd(long_trial_kinematics, axis=0) / np.sqrt(len(long_trial_kinematics))
    else:
        mean_long, sem_long = np.full(TIME_WINDOW_FRAMES * 2, np.nan), np.full(TIME_WINDOW_FRAMES * 2, np.nan)

    time_axis = np.linspace(-TIME_WINDOW_SECONDS/2, TIME_WINDOW_SECONDS/2, len(mean_short))

    if len(short_trial_kinematics) > 0: mean_short = pd.Series(mean_short).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values
    if len(long_trial_kinematics) > 0: mean_long = pd.Series(mean_long).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values

    plt.figure(figsize=(10, 6))
    plt.title(f"Preliminary Data: Pupil Area (Arousal) (Aligned to {event_name})", fontsize=16)
    plt.xlabel(f"Time from {event_name} (seconds)", fontsize=12)
    plt.ylabel(f"Pupil Area (pixels^2)", fontsize=12)

    plt.plot(time_axis, mean_long, color='red', label=f'Long IPI Trials (n={len(long_trial_kinematics)})')
    plt.fill_between(time_axis, mean_long - sem_long, mean_long + sem_long, color='red', alpha=0.2)
    plt.plot(time_axis, mean_short, color='blue', label=f'Short IPI Trials (n={len(short_trial_kinematics)})')
    plt.fill_between(time_axis, mean_short - sem_short, mean_short + sem_short, color='blue', alpha=0.2)
    plt.axvline(0, color='black', linestyle='--')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if 'VisStim' in event_name: event_label = 'Stimulus Onset (0.0s)'
    else: event_label = f'{event_name} (0.0s)'
    handles.extend([
        Line2D([0], [0], color='black', linestyle='--'),
        Line2D([0], [0], marker='o', color='none', mfc='black', mec='black', ms=10),
        Line2D([0], [0], marker='X', color='none', mfc='black', mec='black', ms=10)
    ])
    labels.extend([f'Aligned to {event_name} Event', 'Start (-1.0s)', event_label])
    
    start_index, event_index = 0, TIME_WINDOW_FRAMES
    if not np.all(np.isnan(mean_long)):
        plt.plot(time_axis[start_index], mean_long[start_index], 'o', c='black', mfc='red', ms=10)
        plt.plot(time_axis[event_index], mean_long[event_index], 'X', c='black', mfc='red', ms=12)
    if not np.all(np.isnan(mean_short)):
        plt.plot(time_axis[start_index], mean_short[start_index], 'o', c='black', mfc='blue', ms=10)
        plt.plot(time_axis[event_index], mean_short[event_index], 'X', c='black', mfc='blue', ms=12)

    by_label = dict(zip(labels, handles))
    plt.legend(handles=by_label.values(), labels=by_label.keys())
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = f'my_preliminary_figure_PUPIL_AREA_{event_name}.png'
    plt.savefig(output_filename); plt.close()


# --- 3. LOAD PUPIL/EYE DATA ---
# --- (UNCHANGED) ---

global dlc_data # Make sure plotting functions can see this
try:
    print(f"Loading Sana's processed file: {SANA_PROCESSED_FILE}")
    sana_data = pd.read_csv(SANA_PROCESSED_FILE)
    
    columns = pd.MultiIndex.from_tuples([
        ('PupilCenter', 'x'), ('PupilCenter', 'y'), ('PupilCenter', 'likelihood'),
        ('PupilArea', 'area'), ('PupilArea', 'likelihood')
    ], names=['bodyparts', 'coords'])
    
    dlc_data = pd.DataFrame(index=sana_data.index, columns=columns)
    dlc_data[('PupilCenter', 'x')] = sana_data['Center_X']
    dlc_data[('PupilCenter', 'y')] = sana_data['Center_Y']
    dlc_data[('PupilCenter', 'likelihood')] = 1.0 
    dlc_data[('PupilArea', 'area')] = sana_data['Pupil_Area']
    dlc_data[('PupilArea', 'likelihood')] = 1.0 
    print(f"Successfully loaded and restructured Sana's data. Total frames: {len(dlc_data)}")
    
except FileNotFoundError:
    print(f"--- ERROR: Could not find Sana's file: {SANA_PROCESSED_FILE} ---")
    exit()
except Exception as e:
    print(f"--- ERROR: Could not load Sana's file. Error: {e} ---")
    exit()


# --- 4. LOAD BEHAVIORAL DATA ---
# --- !! REWRITTEN to use bpod_session_data.mat !! ---

print(f"Loading MATLAB file from: {EVENTS_FILE_PATH}")
mat_data = sio.loadmat(EVENTS_FILE_PATH, simplify_cells=True)
print("MAT file loaded. Attempting to parse...")

event_frames = {
    'Push1': [], 'Push2': [], 'VisStim1': [], 'VisStim2': [], 'Reward': [], 'Punish': []
}
trial_conditions = {
    'Push1': [], 'Push2': [], 'VisStim1': [], 'VisStim2': [], 'Reward': [], 'Punish': []
}

# --- !! This is our "best guess" key mapping from your ORIGINAL script !! ---
# --- We may need to debug this if it fails ---
EVENT_NAME_MAP = {
    'Push1': 'Port2In',         # Guess
    'Push2': 'Port2In',         # Guess
    'VisStim1': 'SoftCode1',    # Guess
    'VisStim2': 'SoftCode2',    # Guess
    'Reward': 'Port1Out',       # New guess based on scout output
    'Punish': 'GlobalTimer10_End' # Guess
}

try:
    nTrials = mat_data['SessionData']['nTrials']
    trial_types = mat_data['SessionData']['TrialTypes'] # Our key for short/long
    print(f"Found {nTrials} trials. Looping...")
    
    for i in range(nTrials):
        trial_events = mat_data['SessionData']['RawEvents']['Trial'][i]['Events']
        current_trial_type = trial_types[i]
        is_short = (current_trial_type == TRIAL_TYPE_SHORT)
        
        # --- Handle Push1 and Push2 (special case) ---
        key = EVENT_NAME_MAP['Push1']
        if key in trial_events:
            events = np.atleast_1d(trial_events[key]) 
            if len(events) > 0 and not np.isnan(events[0]):
                event_frame = int(events[0] * VIDEO_FPS)
                event_frames['Push1'].append(event_frame)
                trial_conditions['Push1'].append(is_short)
            if len(events) > 1 and not np.isnan(events[1]):
                event_frame = int(events[1] * VIDEO_FPS)
                event_frames['Push2'].append(event_frame)
                trial_conditions['Push2'].append(is_short)
        
        # --- Handle all other events ---
        for event_name in ['VisStim1', 'VisStim2', 'Reward', 'Punish']:
            key = EVENT_NAME_MAP[event_name]
            if key in trial_events:
                events = np.atleast_1d(trial_events[key])
                if len(events) > 0 and not np.isnan(events[0]):
                    event_frame = int(events[0] * VIDEO_FPS)
                    event_frames[event_name].append(event_frame)
                    trial_conditions[event_name].append(is_short)

    print("Event search complete.")

except KeyError as e:
    print("\n--- SCRIPT ERROR (KeyError) ---")
    print(f"Could not find the key {e}.")
    print("This means the EVENT_NAME_MAP is wrong for this .mat file.")
    print("We need to find the *real* event names.")
    
    # --- Debugging: Print all available event keys from Trial 0 ---
    try:
        print("\n--- DEBUG: Event keys found in Trial 0 ---")
        all_event_keys = mat_data['SessionData']['RawEvents']['Trial'][0]['Events'].keys()
        print(list(all_event_keys))
    except Exception as de:
        print(f"Could not print debug keys: {de}")
    print("---------------------------------\n")
    exit()
except Exception as e:
    print(f"\n--- SCRIPT ERROR (General) ---")
    print(f"An unexpected error occurred: {e}")
    print("---------------------------------\n")
    exit()

# --- 5. RUN ANALYSIS FOR ALL FOUND EVENTS ---
# --- (UNCHANGED) ---

print("\n--- Starting Analysis & Plotting ---")
trial_types_found = False
for event_name, frames in event_frames.items():
    if frames:
        trial_types_found = True
        print(f"Plotting for event: {event_name}")
        
        for coord in COORDINATES_TO_PLOT:
            plot_aligned_data(
                dlc_data[BODYPART_TO_PLOT][coord].copy(),
                frames,
                trial_conditions[event_name],
                True, 
                BODYPART_TO_PLOT,
                coord,
                event_name
            )
        
        plot_trajectory_data(
            frames,
            trial_conditions[event_name],
            BODYPART_TO_PLOT,
            event_name
        )
        
        plot_pupil_area(
            frames,
            trial_conditions[event_name],
            event_name
        )
            
    else:
        print(f"Skipping plot for '{event_name}': No valid events found.")

if not trial_types_found:
    print("\n--- FINAL WARNING ---")
    print("Could not find *any* valid trial data. This likely means the")
    print("EVENT_NAME_MAP in Section 4 is completely wrong for this session.")
    print("Please check the 'DEBUG: Event keys' printout (if it appeared).")

print("\nAll plotting complete.")