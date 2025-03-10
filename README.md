# CERD_Model (Work in Progress)

## Overview
The **Human POV Manipulation Dataset** is a low-cost, real-world dataset for **robotics manipulation learning**. It is captured via a **head-mounted ZED stereo camera**, synchronizing **RGB, depth, and motion data** for **robotics learning, imitation learning, and vision-based control**.

## CERD Pipeline
To efficiently process and structure the dataset, we utilize the **CERD Pipeline**, which ensures high-quality data collection, preprocessing, and organization for machine learning applications.

### 1. Collect Video
Capture video using a **ZED stereo camera**.

### 2. Crop Video
Use `recordings/svo_crop.py` to extract the required portion of the recording.

### 3. Process Frames
Run `processing/vision_data_processing.py` to extract frames and generate:
- **Depth files** (`.npy`)
- **RGB images** (`.png`)
- **Full video** (`.mp4`)

### 4. Set Up GroundedSAM2 & Hamer
Copy `gsam2_run.py` and `hamer_run.py` to their respective directories.

### 5. Run GroundedSAM2
Execute `gsam2_run.py` in the **Grounded-SAM-2** environment to generate:
- **Masks** (`.png`) – for visualizing object segmentation
- **Point clouds** (`.ply`) – for training and computing grasp states

### 6. Run Hamer
Execute `hamer_run.py` in the **Hamer** environment to extract:
- **Hand/joints data** (`.csv`, `.npy`)
- **Hand orientation data** (`.csv`, `.npy`)
- **Visual representations**

### 7. Compute Grasp State
Run `utils/grasp_utils/grasp.py` to generate `grasp.csv`, which contains grasp state information per frame.

### 8. Generate Dataset CSV
Run `dataset_csv.py` to compile all frames into `data.csv`:
```
frame_id,sequence,timestamp,frame,hand_x,hand_y,hand_z,
hand_qx,hand_qy,hand_qz,hand_qw,object_x,object_y,object_z,
grasp_state,total_distance_mm
```

### 9. Extract Keyframes
Run `keyframes_generate.py` to produce:
- **`keyframe_only.csv`** – Extracted keyframes information
- **`smoothed_data.csv`** – Smoothed trajectories based on extracted keyframes

## Dataset Information
```json
{
    "dataset_name": "Human POV Manipulation Dataset",
    "version": "1.0.0",
    "description": "This dataset contains first-person perspective videos and sensor data of human hand manipulation tasks. Data is captured using a head-mounted ZED stereo camera that provides synchronized RGB, depth, and motion information. It is ideal for research in learning-based robotics manipulation, imitation learning, and vision-based control.",
    "contact": {
      "name": "Archer Lin",
      "email": "lin1524@purdue.edu",
      "institution": "Purdue University"
    },
    "citation": "If you use this dataset in your work, please cite: [ Citation Placeholder ].",
    "license": "CC BY 4.0"
}
```

## Data Format and Structure
Each session is stored in a subfolder with RGB images, depth maps, videos, and computed data.

### File Structure
```
📂 dataset/
 ├── 📄 calibration.json       # Camera calibration parameters
 ├── 📂 Task_Name_SessionID/
 │   ├── 📂 rgb_left/          # RGB images from the left camera (1280x720)
 │   ├── 📂 depth/             # Depth maps corresponding to RGB frames
 │   ├── 📂 video/             # MP4 video of the manipulation task
 │   ├── 📂 hand/              # Hand meshes with joints, orientation, and visuals
 │   ├── 📂 masks/             # Filtered object masks (.png) ordered by frames
 │   ├── 📂 point_clouds/      # Filtered object point clouds (.ply) ordered by frames
 │   ├── 📄 data.csv           # Raw data of combined hand trajectory, object position, and grasp_state
 │   ├── 📄 grasp.csv          # Grasp state for each frames
 │   ├── 📄 keyframe_only.csv  # Extracted keyframes data
 │   ├── 📄 smoothed_data.csv  # Smoothed trajectory based on extracted keyframes
 │   ├── 📄 world_pos.csv      # Computed world position of hands and objects

```

### Metadata Fields
- **info.json**
  - `task_name`, `task_description`, `object_used`, `duration`, `environment`, `annotations`
- **result.csv**
  - `sequence_id`, `timestamp`, `hand_position_x/y/z`, `hand_rotation_x/y/z`, `hand_grasp_state`
  - `object_position_x/y/z`, `object_rotation_x/y/z`
  - `camera_position_x/y/z`, `camera_orientation_x/y/z/w`
  - `world_position_x/y/z`, `world_orientation_x/y/z/w`
- **calibration.json**
  - `intrinsic_left`, `intrinsic_right`, `distortion_coefficients_left`, `distortion_coefficients_right`, `extrinsic_matrix`

## Acquisition Details
- **Camera**: ZED Stereo Camera
- **Mounting**: Head-mounted
- **Capture Period**: Start Date: 2025-02-15 | End Date: TBD
- **Environment**: Indoor & Outdoor

## ZED Stereo Camera Video Processing
Since we use the **ZED Stereo Camera**, the **ZED Python API** is required to process **SVO** (Stereo Video Object) files.

### Installation
1. Download and install **ZED SDK**: [ZED SDK Download](https://www.stereolabs.com/developers/release)
2. Install **ZED Python API**: [Python API Installation](https://www.stereolabs.com/docs/app-development/python/install)

### Recording Video
```bash
python svo_recording.py --output_svo_file "C:\Users\arche\OneDrive\Desktop\cerd_videos\Raw_SVO\[name].svo2"
```

### Playback Video
```bash
python svo_playback.py --input_svo_file "C:\Users\arche\OneDrive\Desktop\cerd_videos\Data\[name].svo2"
```

## License
This dataset is released under the **CC BY 4.0 License**. You are free to use, share, and adapt it as long as proper credit is given.

## Citation
If you use this dataset, please cite:
```
[ Citation Placeholder ]
```

