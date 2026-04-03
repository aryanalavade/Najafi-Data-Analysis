import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D # Import for custom legend

# --- 1. CONFIGURATION ---
# --- This file has your PupilCenter (Movement) data ---
PUPIL_CENTER_DLC_FILE = r'/Users/aryan/Downloads/SA09_20251013_Joystick_CG_DB_cam0_run000_20251013_093320DLC_Resnet50_JoystickSA09Oct30shuffle1_snapshot_best-10.csv'

# --- !! NEW: EDIT THIS PATH !! ---
# --- Path to the Pupil Area file you downloaded from the shared drive ---
PUPIL_AREA_DLC_FILE = r'/Users/aryan/Downloads/THE_FILENAME_YOU_FOUND.csv' # <-- EDIT THIS

# --- This file has your bpod (event) data ---
EVENTS_FILE_PATH = r'/Users/aryan/Downloads/SA09_20251013/bpod_session_data.mat' 

BODYPART_TO_PLOT = 'PupilCenter'
COORDINATES_TO_PLOT = ['x', 'y'] 
VIDEO_FPS = 30
TIME_WINDOW_SECONDS = 2
TIME_WINDOW_FRAMES = int(TIME_WINDOW_SECONDS * VIDEO_FPS / 2)
LIKELIHOOD_THRESHOLD = 0.9 
SMOOTHING_WINDOW = 5 

# --- 2. DEFINE PLOTTING FUNCTIONS ---

def plot_aligned_data(kinematic_data, event_frames, is_short_trial, trial_types_found, bodypart, coordinate, event_name):
    """Plots kinematic data aligned to an event (Time Series Plot)."""
    
    kinematic_data[dlc_data_center[bodypart]['likelihood'] < LIKELIHOOD_THRESHOLD] = np.nan
    
    short_trial_kinematics = []
    long_trial_kinematics = []

    for frame_index, is_short in zip(event_frames, is_short_trial):
        start_frame = frame_index - TIME_WINDOW_FRAMES
        end_frame = frame_index + TIME_WINDOW_FRAMES
        
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

    if len(short_trial_kinematics) > 0:
        mean_short = pd.Series(mean_short).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values
    if len(long_trial_kinematics) > 0:
        mean_long = pd.Series(mean_long).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values

    plt.figure(figsize=(10, 6))
    
    # --- !! MODIFIED TITLE !! ---
    plot_title = f"{bodypart} '{coordinate}' Position Aligned to {event_name}"
    plt.title(plot_title, fontsize=16)
    # --- !! ---------------- !! ---
    
    plt.xlabel(f"Time from {event_name} (seconds)", fontsize=12)
    plt.ylabel(f"{bodypart} '{coordinate}' Position (pixels)", fontsize=12)

    # Plot Long IPI
    plt.plot(time_axis, mean_long, color='red', label=f'Long IPI Trials (n={len(long_trial_kinematics)})')
    plt.fill_between(time_axis, mean_long - sem_long, mean_long + sem_long, color='red', alpha=0.2)
    
    # Plot Short IPI
    plt.plot(time_axis, mean_short, color='blue', label=f'Short IPI Trials (n={len(short_trial_kinematics)})')
    plt.fill_between(time_axis, mean_short - sem_short, mean_short + sem_short, color='blue', alpha=0.2)
    
    # Add vertical line for event
    plt.axvline(0, color='black', linestyle='--') # Label is now in custom legend
    
    # Add markers per professor's request
    start_index = 0
    event_index = TIME_WINDOW_FRAMES
    
    if not np.all(np.isnan(mean_long)):
        plt.plot(time_axis[start_index], mean_long[start_index], 'o', c='black', mfc='red', ms=10) # No label
        plt.plot(time_axis[event_index], mean_long[event_index], 'X', c='black', mfc='red', ms=12) # No label
    
    if not np.all(np.isnan(mean_short)):
        plt.plot(time_axis[start_index], mean_short[start_index], 'o', c='black', mfc='blue', ms=10)
        plt.plot(time_axis[event_index], mean_short[event_index], 'X', c='black', mfc='blue', ms=12)

    # --- !! BUG FIX: Corrected Legend Logic !! ---
    
    # Get the line handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels() 
    
    # Create the dynamic event label
    if 'VisStim' in event_name:
        event_label = 'Stimulus Onset (0.0s)'
    else:
        event_label = f'{event_name} (0.0s)'

    # Add the custom marker handles AND LABELS
    handles.extend([
        Line2D([0], [0], color='black', linestyle='--'), # Handle for vline
        Line2D([0], [0], marker='o', color='none', mfc='black', mec='black', ms=10), # Handle for Start
        Line2D([0], [0], marker='X', color='none', mfc='black', mec='black', ms=10)  # Handle for Event
    ])
    labels.extend([
        f'Aligned to {event_name} Event',
        'Start (-1.0s)',
        event_label
    ])
    
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    plt.legend(handles=by_label.values(), labels=by_label.keys())
    # --- !! END BUG FIX !! ---
    
    plt.grid(True, linestyle='--', alpha=0.6)

    output_filename = f'my_preliminary_figure_{coordinate}_pos_{event_name}.png'
    plt.savefig(output_filename)
    print(f"  Successfully saved plot: {output_filename}")
    plt.close()

def plot_trajectory_data(event_frames, is_short_trial, bodypart, event_name):
    """Plots kinematic data as a "Time Trajectory" (X vs Y Plot)."""
    
    x_data = dlc_data_center[bodypart]['x'].copy()
    y_data = dlc_data_center[bodypart]['y'].copy()
    likelihood = dlc_data_center[bodypart]['likelihood']
    x_data[likelihood < LIKELIHOOD_THRESHOLD] = np.nan
    y_data[likelihood < LIKELIHOOD_THRESHOLD] = np.nan

    short_x, short_y, long_x, long_y = [], [], [], []

    for frame_index, is_short in zip(event_frames, is_short_trial):
        start_frame = frame_index - TIME_WINDOW_FRAMES
        end_frame = frame_index + TIME_WINDOW_FRAMES
        
        if start_frame > 0 and end_frame < len(x_data):
            if is_short:
                short_x.append(x_data.iloc[start_frame:end_frame].values)
                short_y.append(y_data.iloc[start_frame:end_frame].values)
            else:
                long_x.append(x_data.iloc[start_frame:end_frame].values)
                long_y.append(y_data.iloc[start_frame:end_frame].values)

    if len(short_x) > 0:
        mean_short_x = np.nanmean(short_x, axis=0)
        mean_short_y = np.nanmean(short_y, axis=0)
    else:
        mean_short_x, mean_short_y = [], []

    if len(long_x) > 0:
        mean_long_x = np.nanmean(long_x, axis=0)
        mean_long_y = np.nanmean(long_y, axis=0)
    else:
        mean_long_x, mean_long_y = [], []

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
    if t_len == 0:
        t_len = len(mean_long_x) if len(mean_long_x) > 0 else len(mean_short_x)
        
    t = np.linspace(-TIME_WINDOW_SECONDS/2, TIME_WINDOW_SECONDS/2, t_len)
    norm = plt.Normalize(t.min(), t.max())
    cmap = 'viridis' 

    def colorline(ax, x, y, t, cmap, linestyle):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles=linestyle)
        lc.set_array(t)
        lc.set_linewidth(3) 
        line = ax.add_collection(lc)
        return line

    # --- Plot 1: Short IPI Trajectory ---
    if len(mean_short_x) > 0:
        colorline(ax1, mean_short_x, mean_short_y, t, cmap, 'solid')
        ax1.plot(mean_short_x[0], mean_short_y[0], 'o', c='black', mfc='white', ms=10) # Start
        ax1.plot(mean_short_x[TIME_WINDOW_FRAMES], mean_short_y[TIME_WINDOW_FRAMES], 'X', c='black', mfc='white', ms=12) # Event
        ax1.set_title('Short IPI Trajectory', fontsize=14)
    ax1.set_xlabel(f"{bodypart} 'x' Position (pixels)", fontsize=12)
    ax1.set_ylabel(f"{bodypart} 'y' Position (pixels)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_aspect('equal', adjustable='box') 

    # --- Plot 2: Long IPI Trajectory ---
    if len(mean_long_x) > 0:
        colorline(ax2, mean_long_x, mean_long_y, t, cmap, 'solid')
        ax2.plot(mean_long_x[0], mean_long_y[0], 'o', c='black', mfc='white', ms=10) # Start
        ax2.plot(mean_long_x[TIME_WINDOW_FRAMES], mean_long_y[TIME_WINDOW_FRAMES], 'X', c='black', mfc='white', ms=12) # Event
        ax2.set_title('Long IPI Trajectory', fontsize=14)
    ax2.set_xlabel(f"{bodypart} 'x' Position (pixels)", fontsize=12)
    ax2.set_ylabel(f"{bodypart} 'y' Position (pixels)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_aspect('equal', adjustable='box') 

    # --- !! MODIFIED TITLE !! ---
    fig.suptitle(f"{bodypart} X-Y Trajectory Aligned to {event_name}", fontsize=18)
    # --- !! ---------------- !! ---
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax1, ax2], location='right', shrink=0.8)
    cbar.set_label('Time from Event (seconds)')

    if 'VisStim' in event_name:
        event_label = 'Stimulus Onset (0.0s)'
    else:
        event_label = f'{event_name} (0.0s)'

    legend_elements = [
        Line2D([0], [0], marker='o', color='none', mfc='black', mec='black', ms=10, label='Start (-1.0s)'),
        Line2D([0], [0], marker='X', color='none', mfc='black', mec='black', ms=10, label=event_label)
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.1, 0.9))

    output_filename = f'my_preliminary_figure_TRAJECTORY_{event_name}.png'
    plt.savefig(output_filename)
    print(f"  Successfully saved plot: {output_filename}")
    plt.close()

# --- !! NEW FUNCTION !! ---
def plot_pupil_area(event_frames, is_short_trial, event_name):
    """Plots pupil area (arousal) aligned to an event."""
    
    # --- !! THIS IS A GUESS - YOU MUST VERIFY !! ---
    # --- Check the 'head' of your new file to confirm these names ---
    BODYPART_NAMES = {
        'top': 'PupilTop',
        'bottom': 'PupilBot', # Often 'PupilBottom' or 'PupilBot'
        'left': 'PupilLeft',
        'right': 'PupilRight'
    }
    # --- !! ------------------------------------ !! ---
    
    try:
        pupil_top = dlc_data_area[BODYPART_NAMES['top']]['y']
        pupil_bottom = dlc_data_area[BODYPART_NAMES['bottom']]['y']
        pupil_left = dlc_data_area[BODYPART_NAMES['left']]['x']
        pupil_right = dlc_data_area[BODYPART_NAMES['right']]['x']
        
        diameter_y = np.abs(pupil_bottom - pupil_top)
        diameter_x = np.abs(pupil_right - pupil_left)
        pupil_area = (diameter_y / 2) * (diameter_x / 2) * np.pi
        
        pupil_area[dlc_data_area[BODYPART_NAMES['top']]['likelihood'] < LIKELIHOOD_THRESHOLD] = np.nan
        pupil_area[dlc_data_area[BODYPART_NAMES['bottom']]['likelihood'] < LIKELIHOOD_THRESHOLD] = np.nan
        pupil_area[dlc_data_area[BODYPART_NAMES['left']]['likelihood'] < LIKELIHOOD_THRESHOLD] = np.nan
        pupil_area[dlc_data_area[BODYPART_NAMES['right']]['likelihood'] < LIKELIHOOD_THRESHOLD] = np.nan

    except KeyError:
        print(f"  Skipping Pupil Area plot for '{event_name}': Could not find bodyparts")
        print("  (e.g., 'PupilTop', 'PupilBot') in the DLC file. Update script with correct names.")
        return
        
    short_trial_kinematics = []
    long_trial_kinematics = []

    for frame_index, is_short in zip(event_frames, is_short_trial):
        start_frame = frame_index - TIME_WINDOW_FRAMES
        end_frame = frame_index + TIME_WINDOW_FRAMES
        
        if start_frame > 0 and end_frame < len(pupil_area):
            data_snippet = pupil_area.iloc[start_frame:end_frame].values
            if is_short:
                short_trial_kinematics.append(data_snippet)
            else:
                long_trial_kinematics.append(data_snippet)

    print(f"  Found {len(short_trial_kinematics)} short (Area) and {len(long_trial_kinematics)} long (Area) trials.")

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

    if len(short_trial_kinematics) > 0:
        mean_short = pd.Series(mean_short).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values
    if len(long_trial_kinematics) > 0:
        mean_long = pd.Series(mean_long).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values

    plt.figure(figsize=(10, 6))
    
    # --- !! MODIFIED TITLE !! ---
    plot_title = f"Pupil Area (Arousal) Aligned to {event_name}"
    plt.title(plot_title, fontsize=16)
    # --- !! ---------------- !! ---
    
    plt.xlabel(f"Time from {event_name} (seconds)", fontsize=12)
    plt.ylabel(f"Pupil Area (approx. pixels^2)", fontsize=12)

    plt.plot(time_axis, mean_long, color='red', label=f'Long IPI Trials (n={len(long_trial_kinematics)})')
    plt.fill_between(time_axis, mean_long - sem_long, mean_long + sem_long, color='red', alpha=0.2)
    plt.plot(time_axis, mean_short, color='blue', label=f'Short IPI Trials (n={len(short_trial_kinematics)})')
    plt.fill_between(time_axis, mean_short - sem_short, mean_short + sem_short, color='blue', alpha=0.2)
    plt.axvline(0, color='black', linestyle='--') # Label is now in custom legend
    
    # --- !! UPDATED: Create a clear, custom legend !! ---
    
    # Get the line handles
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Create the dynamic event label
    if 'VisStim' in event_name:
        event_label = 'Stimulus Onset (0.0s)'
    else:
        event_label = f'{event_name} (0.0s)'

    # Add the custom marker handles AND LABELS
    handles.extend([
        Line2D([0], [0], color='black', linestyle='--'), # Handle for vline
        Line2D([0], [0], marker='o', color='none', mfc='black', mec='black', ms=10), # Handle for Start
        Line2D([0], [0], marker='X', color='none', mfc='black', mec='black', ms=10)  # Handle for Event
    ])
    labels.extend([
        f'Aligned to {event_name} Event',
        'Start (-1.0s)',
        event_label
    ])
    
    # We also need to add the markers to the plot
    start_index = 0
    event_index = TIME_WINDOW_FRAMES
    
    if not np.all(np.isnan(mean_long)):
        plt.plot(time_axis[start_index], mean_long[start_index], 'o', c='black', mfc='red', ms=10)
        plt.plot(time_axis[event_index], mean_long[event_index], 'X', c='black', mfc='red', ms=12)
    
    if not np.all(np.isnan(mean_short)):
        plt.plot(time_axis[start_index], mean_short[start_index], 'o', c='black', mfc='blue', ms=10)
        plt.plot(time_axis[event_index], mean_short[event_index], 'X', c='black', mfc='blue', ms=12)

    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    plt.legend(handles=by_label.values(), labels=by_label.keys())
    # --- !! END UPDATED SECTION !! ---

    plt.grid(True, linestyle='--', alpha=0.6)

    output_filename = f'my_preliminary_figure_PUPIL_AREA_{event_name}.png'
    plt.savefig(output_filename)
    print(f"  Successfully saved plot: {output_filename}")
    plt.close()


# --- 3. LOAD DLC DATA ---
print(f"Loading Pupil Center DLC file: {PUPIL_CENTER_DLC_FILE}")
dlc_data_center = pd.read_csv(PUPIL_CENTER_DLC_FILE, header=[1, 2], index_col=0)
print(f"Successfully loaded Pupil Center DLC data.")

# --- !! NEW !! ---
# Try to load the Pupil Area data
try:
    print(f"Loading Pupil Area DLC file: {PUPIL_AREA_DLC_FILE}")
    dlc_data_area = pd.read_csv(PUPIL_AREA_DLC_FILE, header=[1, 2], index_col=0)
    print(f"Successfully loaded Pupil Area DLC data.")
    has_pupil_area_data = True
except FileNotFoundError:
    print("--- WARNING: Pupil Area DLC file not found. Skipping area plots. ---")
    print("--- Please update 'PUPIL_AREA_DLC_FILE' path when you have it. ---")
    has_pupil_area_data = False
except Exception as e:
    print(f"--- WARNING: Could not load Pupil Area file. Error: {e} ---")
    has_pupil_area_data = False


# --- 4. LOAD BEHAVIORAL DATA (THE "DETECTIVE" PART) ---
print(f"Loading MATLAB file from: {EVENTS_FILE_PATH}")
mat_data = sio.loadmat(EVENTS_FILE_PATH, simplify_cells=True)
print("MAT file loaded. Here are the top-level keys:")
print(mat_data.keys())

try:
    # --- THIS SECTION IS A PLACEHOLDER ---
    # --- !! REPLACE THIS WITH SANA'S/ Dr. Najafi's CODE !! ---
    # --- (If she provides it. If not, this logic is our best bet) ---
    
    print("Keys inside 'SessionData':")
    print(mat_data['SessionData'].keys())
    
    nTrials = mat_data['SessionData']['nTrials']
    trial_types = mat_data['SessionData']['TrialTypes'] # Our key for short/long
    
    print(f"Found {nTrials} trials.")
    print("Keys inside 'SessionData['RawEvents']':")
    print(mat_data['SessionData']['RawEvents'].keys())

    print("Keys inside the *Events* for Trial 0 (for debugging):")
    all_event_keys = mat_data['SessionData']['RawEvents']['Trial'][0]['Events'].keys()
    print(all_event_keys)

    event_frames = {
        'Push1': [], 'Push2': [], 'VisStim1': [], 'VisStim2': [], 'Reward': [], 'Punish': []
    }
    trial_conditions = {
        'Push1': [], 'Push2': [], 'VisStim1': [], 'VisStim2': [], 'Reward': [], 'Punish': []
    }
    
    EVENT_NAME_MAP = {
        'Push1': 'Port2In',
        'Push2': 'Port2In',
        'VisStim1': 'SoftCode1',
        'VisStim2': 'SoftCode2',
        'Reward': 'Port1In',     # This one will fail (key doesn't exist), and that's OK!
        'Punish': 'GlobalTimer10_End'
    }
    
    TRIAL_TYPE_SHORT = 1 

    print("Searching for events in all trials...")
    
    for i in range(nTrials):
        trial_events = mat_data['SessionData']['RawEvents']['Trial'][i]['Events']
        current_trial_type = trial_types[i]
        
        key = EVENT_NAME_MAP['Push1']
        if key in trial_events:
            events = np.atleast_1d(trial_events[key]) 
            if len(events) > 0:
                event_frames['Push1'].append( (events[0] * VIDEO_FPS).astype(int) )
                trial_conditions['Push1'].append(current_trial_type == TRIAL_TYPE_SHORT)
            if len(events) > 1:
                event_frames['Push2'].append( (events[1] * VIDEO_FPS).astype(int) )
                trial_conditions['Push2'].append(current_trial_type == TRIAL_TYPE_SHORT)
        
        for event_name in ['VisStim1', 'VisStim2', 'Reward', 'Punish']:
            key = EVENT_NAME_MAP[event_name]
            if key in trial_events:
                events = np.atleast_1d(trial_events[key])
                if len(events) > 0:
                    event_frames[event_name].append( (events[0] * VIDEO_FPS).astype(int) )
                    trial_conditions[event_name].append(current_trial_type == TRIAL_TYPE_SHORT)

    print("Event search complete.")
    # --- !! END OF PLACEHOLDER SECTION !! ---

except KeyError as e:
    print(f"\n--- SCRIPT ERROR (KeyError) ---")
    print(f"Could not find the key {e}.")
    print("This is part of the 'detective' process. Look at the printed keys")
    print("and update the logic in Section 4 to match your data structure.")
    print("---------------------------------\n")
    exit()
except Exception as e:
    print(f"\n--- SCRIPT ERROR (General) ---")
    print(f"An unexpected error occurred: {e}")
    print("Please review the script and error message.")
    print("---------------------------------\n")
    exit()

# --- 5. RUN ANALYSIS FOR ALL FOUND EVENTS ---
print("\n--- Starting Analysis & Plotting ---")
trial_types_found = False
for event_name, frames in event_frames.items():
    if frames:
        trial_types_found = True
        print(f"Plotting for event: {event_name}")
        
        # Plot X and Y (Time Series)
        for coord in COORDINATES_TO_PLOT:
            plot_aligned_data(
                dlc_data_center[BODYPART_TO_PLOT][coord].copy(),
                frames,
                trial_conditions[event_name],
                True, 
                BODYPART_TO_PLOT,
                coord,
                event_name
            )
        
        # Plot the X-Y Trajectory
        plot_trajectory_data(
            frames,
            trial_conditions[event_name],
            BODYPART_TO_PLOT,
            event_name
        )
        
        # --- !! NEW !! ---
        # Plot Pupil Area (if we have the data)
        if has_pupil_area_data:
            plot_pupil_area(
                frames,
                trial_conditions[event_name],
                event_name
            )
            
    else:
        print(f"Skipping plot for '{event_name}': No valid events found.")

if not trial_types_found:
    print("\n--- FINAL ERROR ---")
    print("Could not find *any* valid trial data. The event names in EVENT_NAME_MAP are likely all incorrect.")
    print("Please check the 'Keys inside the *Events* for Trial 0' printout and update the EVENT_NAME_MAP.")

print("\nAll plotting complete.")