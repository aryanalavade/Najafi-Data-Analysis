import deeplabcut
import os

config = "/storage/scratch1/8/analavade3/FullBodyTracking-Arya-2025-11-17/config.yaml"

video_folder = "/storage/scratch1/8/analavade3/FullBodyTracking-Arya-2025-11-17/videos"

# Get all .avi videos
all_videos = [f for f in os.listdir(video_folder) if f.endswith('.avi')]

# Get already processed CSVs
processed_csvs = [f for f in os.listdir(video_folder) if f.endswith('.csv')]

# Convert CSV names → corresponding video names
processed_videos = [f.replace('.csv', '.avi') for f in processed_csvs]

# Only keep videos that are NOT processed
videos_to_run = [v for v in all_videos if v not in processed_videos]

print(f"Total videos: {len(all_videos)}")
print(f"Already processed: {len(processed_videos)}")
print(f"Remaining videos: {len(videos_to_run)}")

# Convert to full paths
videos_to_run_paths = [os.path.join(video_folder, v) for v in videos_to_run]

# Run DLC
deeplabcut.analyze_videos(
    config,
    videos_to_run_paths,
    videotype='avi',
    destfolder=video_folder,
    save_as_csv=True
)






# import deeplabcut

# config = "/storage/scratch1/8/analavade3/FullBodyTracking-Arya-2025-11-17/config.yaml"

# videos_folder_path = "/storage/scratch1/8/analavade3/FullBodyTracking-Arya-2025-11-17/videos"

# deeplabcut.analyze_videos(
#     config,
#    [videos_folder_path],
#    videotype='avi',
#   destfolder=videos_folder_path,
#    save_as_csv=True
#)