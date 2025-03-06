import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_hand_trajectory(csv_path):
    """
    Load hand trajectory data from a CSV file.
    
    Expected CSV columns:
      - time: Timestamps
      - x, y, z: End-effector positions
      - qx, qy, qz, qw: Quaternion orientation components
      - gripper_width: Gripper state (0 for open, 1 for closed)
    """
    return pd.read_csv(csv_path)

def compute_dynamics(positions, timestamps):
    """
    Compute velocities and accelerations based on positions.
    
    Parameters:
      positions (np.ndarray): Nx3 array of positions.
      timestamps (np.ndarray): 1D array of time stamps.
    
    Returns:
      velocities, accelerations (np.ndarray): Both are Nx3 arrays.
    """
    velocities = np.gradient(positions, timestamps, axis=0)
    accelerations = np.gradient(velocities, timestamps, axis=0)
    return velocities, accelerations

def identify_key_frames(timestamps, positions, gripper_states,
                        vel_threshold=0.05, acc_threshold=0.02, 
                        gripper_threshold=0.5):
    """
    Identify key frames using local extrema of velocity, acceleration,
    and changes in gripper state.
    
    Parameters:
      timestamps (np.ndarray): Array of time stamps.
      positions (np.ndarray): Nx3 array of positions.
      gripper_states (np.ndarray): 1D array of binary gripper state values (0 or 1).
      vel_threshold (float): Minimum velocity magnitude to trigger a key frame.
      acc_threshold (float): Minimum acceleration magnitude to trigger a key frame.
      gripper_threshold (float): Threshold for detecting a gripper state change.
                                Since states are binary, any change is significant.
      
    Returns:
      key_frames (list): Sorted list of indices corresponding to key frames.
    """
    velocities, accelerations = compute_dynamics(positions, timestamps)
    
    # Compute the magnitude of velocity and acceleration
    vel_norm = np.linalg.norm(velocities, axis=1)
    acc_norm = np.linalg.norm(accelerations, axis=1)
    
    # Find peaks in the velocity and acceleration norms
    vel_peaks, _ = find_peaks(vel_norm, height=vel_threshold)
    acc_peaks, _ = find_peaks(acc_norm, height=acc_threshold)
    
    # Detect gripper state changes (since gripper_states are binary,
    # a change is when the difference is non-zero)
    gripper_diff = np.abs(np.diff(gripper_states, prepend=gripper_states[0]))
    gripper_peaks, _ = find_peaks(gripper_diff, height=gripper_threshold)
    
    # Combine key indices from all criteria
    key_indices = set()
    key_indices.update(vel_peaks.tolist())
    key_indices.update(acc_peaks.tolist())
    key_indices.update(gripper_peaks.tolist())
    
    # Always include the first and last frame
    key_indices.add(0)
    key_indices.add(len(timestamps) - 1)
    
    return sorted(list(key_indices))

def interpolate_quaternions(key_times, key_quaternions, new_times):
    """
    Interpolate quaternion values over new timestamps using SLERP.
    
    Parameters:
      key_times (np.ndarray): Timestamps corresponding to key quaternions.
      key_quaternions (np.ndarray): Nx4 array of key quaternions (qx, qy, qz, qw).
      new_times (np.ndarray): Timestamps for the interpolated trajectory.
      
    Returns:
      np.ndarray: Interpolated quaternions in (qx, qy, qz, qw) format.
    """
    key_rotations = R.from_quat(key_quaternions)
    slerp = Slerp(key_times, key_rotations)
    target_rotations = slerp(new_times)
    return target_rotations.as_quat()

def interpolate_positions(key_times, key_positions, new_times):
    """
    Interpolate positions using cubic spline interpolation.
    
    Parameters:
      key_times (np.ndarray): Timestamps for key positions.
      key_positions (np.ndarray): Nx3 array of key positions.
      new_times (np.ndarray): Timestamps for the interpolated trajectory.
      
    Returns:
      np.ndarray: Interpolated positions.
    """
    cs = CubicSpline(key_times, key_positions, axis=0)
    return cs(new_times)

def interpolate_gripper(key_times, key_gripper, new_times):
    """
    Interpolate the gripper state as a step function.
    For each new time, assign the gripper state from the most recent key frame.
    
    Parameters:
      key_times (np.ndarray): Timestamps for key gripper states.
      key_gripper (np.ndarray): 1D array of binary gripper state values.
      new_times (np.ndarray): Timestamps for the interpolated trajectory.
      
    Returns:
      np.ndarray: Interpolated binary gripper states.
    """
    # For each new time, find the index of the most recent key time
    indices = np.searchsorted(key_times, new_times, side='right') - 1
    indices = np.clip(indices, 0, len(key_gripper)-1)
    return key_gripper[indices]

def interpolate_trajectory(df, 
                           vel_threshold=0.05, acc_threshold=0.02, gripper_threshold=0.5):
    """
    Generate a smoothed trajectory from the original data by:
      1. Selecting key frames based on motion dynamics and gripper state changes.
      2. Interpolating positions, orientations, and gripper states between key frames.
      
    Parameters:
      df (pd.DataFrame): DataFrame with columns ['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper_width'].
      vel_threshold (float): Threshold for velocity peak detection.
      acc_threshold (float): Threshold for acceleration peak detection.
      gripper_threshold (float): Threshold for gripper state change detection.
      
    Returns:
      pd.DataFrame: New DataFrame containing the smoothed trajectory.
    """
    timestamps = df["time"].values
    positions = df[["x", "y", "z"]].values
    quaternions = df[["qx", "qy", "qz", "qw"]].values
    gripper_states = df["gripper_width"].values
    
    # Identify key frames using dynamics and gripper state changes
    key_frame_indices = identify_key_frames(timestamps, positions, gripper_states,
                                            vel_threshold, acc_threshold, gripper_threshold)
    key_times = timestamps[key_frame_indices]
    key_positions = positions[key_frame_indices]
    key_quaternions = quaternions[key_frame_indices]
    key_gripper = gripper_states[key_frame_indices]
    
    # Generate new, evenly spaced timestamps over the original time span
    new_times = np.linspace(timestamps[0], timestamps[-1], num=len(timestamps))
    
    # Interpolate positions, quaternions, and gripper state
    smooth_positions = interpolate_positions(key_times, key_positions, new_times)
    smooth_quaternions = interpolate_quaternions(key_times, key_quaternions, new_times)
    smooth_gripper = interpolate_gripper(key_times, key_gripper, new_times)
    
    # Create a new DataFrame with the smoothed trajectory data
    new_df = pd.DataFrame({
        "time": new_times,
        "x": smooth_positions[:, 0],
        "y": smooth_positions[:, 1],
        "z": smooth_positions[:, 2],
        "qx": smooth_quaternions[:, 0],
        "qy": smooth_quaternions[:, 1],
        "qz": smooth_quaternions[:, 2],
        "qw": smooth_quaternions[:, 3],
        "gripper_width": smooth_gripper
    })
    
    return new_df

if __name__ == "__main__":
    input_csv = "hand_trajectory.csv"  # Update this path to your CSV file
    output_csv = "smoothed_hand_trajectory.csv"
    
    # Load the original hand trajectory data
    df = load_hand_trajectory(input_csv)
    
    # Generate the smoothed trajectory using refined keyframe criteria
    smoothed_df = interpolate_trajectory(df,
                                         vel_threshold=0.05,
                                         acc_threshold=0.02,
                                         gripper_threshold=0.5)
    
    # Save the smoothed trajectory to a new CSV file
    smoothed_df.to_csv(output_csv, index=False)
    print(f"Smoothed trajectory saved to {output_csv}")
    
    # Plot a comparison of original vs. smoothed X positions for verification
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["x"], label="Original X", alpha=0.5)
    plt.plot(smoothed_df["time"], smoothed_df["x"], label="Smoothed X", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Position X")
    plt.title("Hand Position: Original vs. Smoothed")
    plt.legend()
    plt.show()
