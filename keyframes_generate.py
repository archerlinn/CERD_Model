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

def identify_key_frames(frame_ids, positions, grasp_states,
                        vel_threshold=0.05, acc_threshold=0.02, grasp_threshold=0.5):
    """
    Identify key frame indices using local extrema in velocity, acceleration,
    and changes in grasp state, with respect to frame_id.
    
    This function finds both peaks and troughs for velocity and acceleration.
    
    Parameters:
      frame_ids (np.ndarray): 1D array of frame IDs.
      positions (np.ndarray): Nx3 array of hand positions.
      grasp_states (np.ndarray): 1D array of grasp state values.
      vel_threshold (float): Minimum magnitude to consider a keyframe for velocity.
      acc_threshold (float): Minimum magnitude to consider a keyframe for acceleration.
      grasp_threshold (float): Threshold for detecting a change in grasp state.
    
    Returns:
      Sorted list of indices that are key frames.
    """
    velocities, accelerations = compute_dynamics(positions, frame_ids)
    vel_norm = np.linalg.norm(velocities, axis=1)
    acc_norm = np.linalg.norm(accelerations, axis=1)
    
    # Find local peaks (maxima) and troughs (minima) for velocity and acceleration:
    vel_peaks, _ = find_peaks(vel_norm, height=vel_threshold)
    vel_troughs, _ = find_peaks(-vel_norm, height=vel_threshold)
    acc_peaks, _ = find_peaks(acc_norm, height=acc_threshold)
    acc_troughs, _ = find_peaks(-acc_norm, height=acc_threshold)
    
    # Detect grasp state changes
    grasp_diff = np.abs(np.diff(grasp_states, prepend=grasp_states[0]))
    grasp_peaks, _ = find_peaks(grasp_diff, height=grasp_threshold)
    
    key_indices = (set(vel_peaks.tolist()) | set(vel_troughs.tolist()) |
                   set(acc_peaks.tolist()) | set(acc_troughs.tolist()) |
                   set(grasp_peaks.tolist()))
    # Always include first and last frame
    key_indices.add(0)
    key_indices.add(len(frame_ids)-1)
    
    return sorted(list(key_indices))

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
    indices = np.clip(indices, 0, len(key_values)-1)
    return key_values[indices]

def smooth_trajectory(df, vel_threshold=0.05, acc_threshold=0.02, grasp_threshold=0.5):
    """
    Generate a fully smoothed trajectory using keyframe-based interpolation.
    The interpolation is computed over the entire frame_id grid.
    
    Returns a new DataFrame with the same number of rows and same columns as the input.
    """
    # Use frame_id as the independent variable.
    frame_ids = df["frame_id"].values.astype(float)
    hand_positions = df[["hand_x", "hand_y", "hand_z"]].values.astype(float)
    hand_quaternions = df[["hand_qx", "hand_qy", "hand_qz", "hand_qw"]].values.astype(float)
    obj_positions = df[["object_x", "object_y", "object_z"]].values.astype(float)
    grasp_states = df["grasp_state"].values.astype(float)
    total_distance = df["total_distance_mm"].values.astype(float)
    
    # Identify key frame indices
    key_indices = identify_key_frames(frame_ids, hand_positions, grasp_states,
                                      vel_threshold, acc_threshold, grasp_threshold)
    
    key_times = frame_ids[key_indices]
    key_hand_positions = hand_positions[key_indices]
    key_hand_quaternions = hand_quaternions[key_indices]
    key_obj_positions = obj_positions[key_indices]
    key_grasp = grasp_states[key_indices]
    key_total_distance = total_distance[key_indices]
    
    new_times = frame_ids  # Use the original frame_id grid for interpolation
    
    # Interpolate each component fully from the key frames:
    smooth_hand_positions = interpolate_positions(key_times, key_hand_positions, new_times)
    smooth_hand_quaternions = interpolate_quaternions(key_times, key_hand_quaternions, new_times)
    smooth_obj_positions = interpolate_positions(key_times, key_obj_positions, new_times)
    smooth_grasp = interpolate_step(key_times, key_grasp, new_times)
    smooth_total_distance = interpolate_step(key_times, key_total_distance, new_times)
    
    # Build a new DataFrame preserving the original columns
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

def extract_keyframes(df, vel_threshold=0.05, acc_threshold=0.02, grasp_threshold=0.5):
    """
    Extract only the key frame rows from the DataFrame based on keyframe criteria.
    """
    frame_ids = df["frame_id"].values.astype(float)
    hand_positions = df[["hand_x", "hand_y", "hand_z"]].values.astype(float)
    grasp_states = df["grasp_state"].values.astype(float)
    
    key_indices = identify_key_frames(frame_ids, hand_positions, grasp_states,
                                      vel_threshold, acc_threshold, grasp_threshold)
    key_df = df.iloc[key_indices].reset_index(drop=True)
    return key_df

def main():
    input_csv = "/home/archer/cerd_data/pour_water_01/data.csv"            # Path to your input CSV file
    output_csv_smoothed = "/home/archer/cerd_data/pour_water_01/keyframe_data.csv"  # Output CSV file for smoothed trajectory
    output_csv_keyframes = "/home/archer/cerd_data/pour_water_01/keyframe_only.csv" # Output CSV file for key frames only
    
    df = load_data(input_csv)
    
    # Compute fully smoothed trajectory (all frames, interpolated from key frames)
    smoothed_df = smooth_trajectory(df, vel_threshold=0.05, acc_threshold=0.02, grasp_threshold=0.5)
    smoothed_df.to_csv(output_csv_smoothed, index=False)
    print(f"Smoothed trajectory saved to {output_csv_smoothed}")
    
    # Extract only key frames from the original data
    keyframes_df = extract_keyframes(df, vel_threshold=0.05, acc_threshold=0.02, grasp_threshold=0.5)
    keyframes_df.to_csv(output_csv_keyframes, index=False)
    print(f"Key frames saved to {output_csv_keyframes}")
    
    # Optional: Plot a comparison for one coordinate (hand_x) with key frames highlighted.
    plt.figure(figsize=(12,6))
    plt.plot(df["frame_id"], df["hand_x"], label="Original hand_x", alpha=0.5)
    plt.plot(smoothed_df["frame_id"], smoothed_df["hand_x"], label="Smoothed hand_x", linestyle="--")
    plt.scatter(keyframes_df["frame_id"], keyframes_df["hand_x"], color="red", label="Key frames", zorder=5)
    plt.xlabel("Frame ID")
    plt.ylabel("hand_x")
    plt.title("Trajectory: Original vs. Smoothed vs. Key Frames")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
