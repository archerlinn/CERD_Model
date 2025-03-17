#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def load_data(csv_path):
    """
    Load trajectory data from a CSV file.
    
    Expected CSV columns:
      frame_id, sequence, timestamp, frame,
      hand_x, hand_y, hand_z, hand_qx, hand_qy, hand_qz, hand_qw,
      object_x, object_y, object_z, grasp_state, total_distance_mm
    """
    return pd.read_csv(csv_path)

def compute_dynamics(positions, frame_ids):
    """
    Compute velocities and accelerations with respect to frame_id.
    """
    velocities = np.gradient(positions, frame_ids, axis=0)
    accelerations = np.gradient(velocities, frame_ids, axis=0)
    return velocities, accelerations

def filter_keyframes_by_gap(key_indices, min_gap):
    """
    Filter keyframe indices so that any two consecutive indices are at least
    min_gap apart.
    """
    if not key_indices:
        return key_indices
    filtered = [key_indices[0]]
    last = key_indices[0]
    for idx in key_indices[1:]:
        if idx - last >= min_gap:
            filtered.append(idx)
            last = idx
    return filtered

def supplement_keyframes(key_indices, total_frames, max_key_frames, min_gap):
    """
    If there are fewer than max_key_frames, supplement with evenly spaced indices
    from the full trajectory while ensuring a minimum gap.
    """
    even_indices = np.linspace(0, total_frames - 1, max_key_frames, dtype=int).tolist()
    combined = sorted(set(key_indices + even_indices))
    combined = filter_keyframes_by_gap(combined, min_gap)
    if len(combined) > max_key_frames:
        idxs = np.linspace(0, len(combined) - 1, max_key_frames, dtype=int)
        combined = [combined[i] for i in idxs]
    return combined

def identify_key_frames(frame_ids, positions, grasp_states,
                        vel_threshold=0.05, acc_threshold=0.02, grasp_threshold=0.5,
                        max_key_frames=15, min_gap=10):
    """
    Identify key frame indices using local extrema in the raw velocity and acceleration
    signals, and changes in the grasp state.
    
    Returns:
      Sorted list of key frame indices.
    """
    velocities, accelerations = compute_dynamics(positions, frame_ids)
    vel_norm = np.linalg.norm(velocities, axis=1)
    acc_norm = np.linalg.norm(accelerations, axis=1)
    
    # Detect peaks and troughs.
    vel_peaks, _ = find_peaks(vel_norm, height=vel_threshold)
    vel_troughs, _ = find_peaks(-vel_norm, height=vel_threshold)
    acc_peaks, _ = find_peaks(acc_norm, height=acc_threshold)
    acc_troughs, _ = find_peaks(-acc_norm, height=acc_threshold)
    
    # Detect grasp state changes.
    grasp_diff = np.abs(np.diff(grasp_states, prepend=grasp_states[0]))
    grasp_peaks, _ = find_peaks(grasp_diff, height=grasp_threshold)
    
    # Combine all detected indices and always include the first and last frame.
    key_indices = set(vel_peaks.tolist() + vel_troughs.tolist() +
                      acc_peaks.tolist() + acc_troughs.tolist() +
                      grasp_peaks.tolist())
    key_indices.add(0)
    key_indices.add(len(frame_ids) - 1)
    
    key_indices = sorted(list(key_indices))
    key_indices = filter_keyframes_by_gap(key_indices, min_gap)
    key_indices = supplement_keyframes(key_indices, len(frame_ids), max_key_frames, min_gap)
    
    return key_indices

def interpolate_quaternions(key_times, key_quaternions, new_times):
    """Interpolate quaternions using SLERP."""
    key_rotations = R.from_quat(key_quaternions)
    slerp = Slerp(key_times, key_rotations)
    target_rotations = slerp(new_times)
    return target_rotations.as_quat()

def interpolate_positions(key_times, key_positions, new_times):
    """Interpolate positions using cubic spline interpolation."""
    cs = CubicSpline(key_times, key_positions, axis=0)
    return cs(new_times)

def interpolate_step(key_times, key_values, new_times):
    """Interpolate step-like values using zero-order hold."""
    indices = np.searchsorted(key_times, new_times, side='right') - 1
    indices = np.clip(indices, 0, len(key_values) - 1)
    return key_values[indices]

def smooth_trajectory(df, vel_threshold=0.05, acc_threshold=0.02, grasp_threshold=0.5,
                      max_key_frames=15, min_gap=10, user_exclude=None):
    """
    Generate a fully smoothed trajectory using keyframe-based interpolation.
    Optionally remove keyframes (specified by their 1-indexed order) if they are outliers.
    
    Returns a new DataFrame with the same columns as the input.
    """
    frame_ids = df["frame_id"].values.astype(float)
    hand_positions = df[["hand_x", "hand_y", "hand_z"]].values.astype(float)
    hand_quaternions = df[["hand_qx", "hand_qy", "hand_qz", "hand_qw"]].values.astype(float)
    obj_positions = df[["object_x", "object_y", "object_z"]].values.astype(float)
    grasp_states = df["grasp_state"].values.astype(float)
    total_distance = df["total_distance_mm"].values.astype(float)
    
    key_indices = identify_key_frames(frame_ids, hand_positions, grasp_states,
                                      vel_threshold, acc_threshold, grasp_threshold,
                                      max_key_frames=max_key_frames, min_gap=min_gap)
    # Remove keyframes based on user input (user_exclude holds 1-indexed positions)
    if user_exclude is not None:
        key_indices = [k for i, k in enumerate(key_indices, start=1) if i not in user_exclude]
    
    key_times = frame_ids[key_indices]
    key_hand_positions = hand_positions[key_indices]
    key_hand_quaternions = hand_quaternions[key_indices]
    key_obj_positions = obj_positions[key_indices]
    key_grasp = grasp_states[key_indices]
    key_total_distance = total_distance[key_indices]
    
    new_times = frame_ids  # Interpolate over the entire frame_id grid.
    
    smooth_hand_positions = interpolate_positions(key_times, key_hand_positions, new_times)
    smooth_hand_quaternions = interpolate_quaternions(key_times, key_hand_quaternions, new_times)
    smooth_obj_positions = interpolate_positions(key_times, key_obj_positions, new_times)
    smooth_grasp = interpolate_step(key_times, key_grasp, new_times)
    smooth_total_distance = interpolate_step(key_times, key_total_distance, new_times)
    
    new_df = pd.DataFrame({
        "frame_id": df["frame_id"],
        "sequence": df["sequence"],
        "timestamp": df["timestamp"],
        "frame": df["frame"],
        "hand_x": smooth_hand_positions[:, 0],
        "hand_y": smooth_hand_positions[:, 1],
        "hand_z": smooth_hand_positions[:, 2],
        "hand_qx": smooth_hand_quaternions[:, 0],
        "hand_qy": smooth_hand_quaternions[:, 1],
        "hand_qz": smooth_hand_quaternions[:, 2],
        "hand_qw": smooth_hand_quaternions[:, 3],
        "object_x": smooth_obj_positions[:, 0],
        "object_y": smooth_obj_positions[:, 1],
        "object_z": smooth_obj_positions[:, 2],
        "grasp_state": smooth_grasp,
        "total_distance_mm": smooth_total_distance
    })
    return new_df

def extract_keyframes(df, vel_threshold=0.05, acc_threshold=0.02, grasp_threshold=0.5,
                      max_key_frames=15, min_gap=10, user_exclude=None):
    """
    Extract only the key frame rows from the DataFrame based on keyframe criteria.
    Optionally remove keyframes specified in user_exclude (1-indexed positions).
    """
    frame_ids = df["frame_id"].values.astype(float)
    hand_positions = df[["hand_x", "hand_y", "hand_z"]].values.astype(float)
    grasp_states = df["grasp_state"].values.astype(float)
    
    key_indices = identify_key_frames(frame_ids, hand_positions, grasp_states,
                                      vel_threshold, acc_threshold, grasp_threshold,
                                      max_key_frames=max_key_frames, min_gap=min_gap)
    if user_exclude is not None:
        key_indices = [k for i, k in enumerate(key_indices, start=1) if i not in user_exclude]
    key_df = df.iloc[key_indices].reset_index(drop=True)
    return key_df

def main():
    input_csv = "/home/archer/cerd_data/pour_water_07/data.csv"            # Path to your input CSV file
    output_csv_smoothed = "/home/archer/cerd_data/pour_water_07/smoothed_data.csv"  # Output CSV for smoothed trajectory
    output_csv_keyframes = "/home/archer/cerd_data/pour_water_07/keyframe_only.csv" # Output CSV for key frames only
    
    df = load_data(input_csv)
    
    # First, compute the initial keyframes for display.
    frame_ids = df["frame_id"].values.astype(float)
    hand_positions = df[["hand_x", "hand_y", "hand_z"]].values.astype(float)
    grasp_states = df["grasp_state"].values.astype(float)
    initial_key_indices = identify_key_frames(frame_ids, hand_positions, grasp_states,
                                               vel_threshold=0.05, acc_threshold=0.02,
                                               grasp_threshold=0.5, max_key_frames=15, min_gap=10)
    
    print("Detected Keyframes:")
    for i, k in enumerate(initial_key_indices, start=1):
        actual_frame = df["frame_id"].iloc[k]
        print(f"{i}: frame number = {actual_frame}")
    print("After you analyzed, close the graphs to enter keyframes")

    # Visualize initial keyframes for hand_x, hand_y, and hand_z.
    plt.figure(figsize=(12,6))
    plt.plot(df["frame_id"], df["hand_x"], label="Original hand_x", alpha=0.5)
    plt.scatter(df.iloc[initial_key_indices]["frame_id"], df.iloc[initial_key_indices]["hand_x"],
                color="red", label="Detected Key Frames", zorder=5)
    plt.xlabel("Frame ID")
    plt.ylabel("hand_x")
    plt.title("Initial Key Frames Detection (hand_x)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(df["frame_id"], df["hand_y"], label="Original hand_y", alpha=0.5)
    plt.scatter(df.iloc[initial_key_indices]["frame_id"], df.iloc[initial_key_indices]["hand_y"],
                color="red", label="Detected Key Frames", zorder=5)
    plt.xlabel("Frame ID")
    plt.ylabel("hand_y")
    plt.title("Initial Key Frames Detection (hand_y)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(df["frame_id"], df["hand_z"], label="Original hand_z", alpha=0.5)
    plt.scatter(df.iloc[initial_key_indices]["frame_id"], df.iloc[initial_key_indices]["hand_z"],
                color="red", label="Detected Key Frames", zorder=5)
    plt.xlabel("Frame ID")
    plt.ylabel("hand_z")
    plt.title("Initial Key Frames Detection (hand_z)")
    plt.legend()
    plt.show()
    
    # Now prompt the user to enter keyframe numbers (1-indexed) to remove.
    user_input = input("Enter the keyframe numbers to remove (comma-separated), or press Enter to keep all: ")
    if user_input.strip():
        try:
            user_exclude = [int(x.strip()) for x in user_input.split(",") if x.strip().isdigit()]
        except Exception as e:
            print("Invalid input, no keyframes will be removed.")
            user_exclude = None
    else:
        user_exclude = None
    
    # Compute the fully smoothed trajectory using keyframes (with any exclusions applied).
    smoothed_df = smooth_trajectory(df, vel_threshold=0.05, acc_threshold=0.02,
                                    grasp_threshold=0.5, max_key_frames=15, min_gap=10,
                                    user_exclude=user_exclude)
    smoothed_df.to_csv(output_csv_smoothed, index=False)
    print(f"Smoothed trajectory saved to {output_csv_smoothed}")
    
    # Extract only keyframes (with exclusions) from the original data.
    keyframes_df = extract_keyframes(df, vel_threshold=0.05, acc_threshold=0.02,
                                     grasp_threshold=0.5, max_key_frames=15, min_gap=10,
                                     user_exclude=user_exclude)
    keyframes_df.to_csv(output_csv_keyframes, index=False)
    print(f"Key frames saved to {output_csv_keyframes}")
    
    # Visualize the final smoothed trajectory and keyframes.
    plt.figure(figsize=(12,6))
    plt.plot(df["frame_id"], df["hand_x"], label="Original hand_x", alpha=0.5)
    plt.plot(smoothed_df["frame_id"], smoothed_df["hand_x"], label="Smoothed hand_x", linestyle="--")
    plt.scatter(keyframes_df["frame_id"], keyframes_df["hand_x"], color="red", label="Key Frames", zorder=5)
    plt.xlabel("Frame ID")
    plt.ylabel("hand_x")
    plt.title("Final Trajectory: Original vs. Smoothed vs. Key Frames (hand_x)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(df["frame_id"], df["hand_y"], label="Original hand_y", alpha=0.5)
    plt.plot(smoothed_df["frame_id"], smoothed_df["hand_y"], label="Smoothed hand_y", linestyle="--")
    plt.scatter(keyframes_df["frame_id"], keyframes_df["hand_y"], color="red", label="Key Frames", zorder=5)
    plt.xlabel("Frame ID")
    plt.ylabel("hand_y")
    plt.title("Final Trajectory: Original vs. Smoothed vs. Key Frames (hand_y)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(df["frame_id"], df["hand_z"], label="Original hand_z", alpha=0.5)
    plt.plot(smoothed_df["frame_id"], smoothed_df["hand_z"], label="Smoothed hand_z", linestyle="--")
    plt.scatter(keyframes_df["frame_id"], keyframes_df["hand_z"], color="red", label="Key Frames", zorder=5)
    plt.xlabel("Frame ID")
    plt.ylabel("hand_z")
    plt.title("Final Trajectory: Original vs. Smoothed vs. Key Frames (hand_z)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
