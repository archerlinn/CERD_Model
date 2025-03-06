"""
Assumptions:

The input CSV has columns for the hand and object in camera coordinates:
For the hand: "hand_x", "hand_y", "hand_z", "hand_qx", "hand_qy", "hand_qz", "hand_qw".
For the object: "object_x", "object_y", "object_z", "object_qx", "object_qy", "object_qz", "object_qw".
The extrinsics JSON defines rotation offsets ("RX", "CV", "RZ") and a translation vector along with a "baseline".
Here we assume:
"RX" rotates about the X axis, "CV" about the Y axis, and "RZ" about the Z axis.
The order of rotations is applied in the 'xyz' order.
The translation vector is given as [tx, ty], and the baseline (converted from mm to m) is used as the Z translation.
You may need to adjust these assumptions based on your calibration details.
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def load_extrinsics(json_file: str) -> np.ndarray:
    """
    Loads the extrinsics from a JSON file and computes a 4x4 homogeneous transformation matrix.
    
    Assumptions:
      - Rotation offsets: 'RX', 'CV', 'RZ' (applied in 'xyz' order).
      - Translation: Provided translation vector [tx, ty] and a 'baseline' used as z (converted to meters).
    
    Parameters:
        json_file (str): Path to the extrinsics JSON file.
    
    Returns:
        np.ndarray: A 4x4 transformation matrix mapping camera coordinates to world coordinates.
    """
    with open(json_file, 'r') as f:
        extrinsics = json.load(f)
    
    # Extract rotation offsets
    rx = extrinsics["extrinsic_matrix"]["rotation_offsets"]["RX"]
    cv = extrinsics["extrinsic_matrix"]["rotation_offsets"]["CV"]
    rz = extrinsics["extrinsic_matrix"]["rotation_offsets"]["RZ"]
    
    # Create rotation using Euler angles (order: X, Y, Z)
    angles = [rx, cv, rz]
    rotation = R.from_euler('xyz', angles)
    
    # Extract translation: translation_vector = [tx, ty] and use baseline as z (convert mm to m)
    tx, ty = extrinsics["extrinsic_matrix"]["translation_vector"]
    baseline_mm = extrinsics["extrinsic_matrix"]["baseline"]
    tz = baseline_mm / 1000.0  # convert millimeters to meters
    
    translation = np.array([tx, ty, tz])
    
    # Construct the homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = translation
    
    return T

def transform_points(camera_points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transforms an array of 3D points from camera to world coordinates using the provided transformation matrix.
    
    Parameters:
        camera_points (np.ndarray): Array of shape (N, 3) with camera coordinates.
        transform_matrix (np.ndarray): 4x4 transformation matrix.
    
    Returns:
        np.ndarray: Array of shape (N, 3) with world coordinates.
    """
    num_points = camera_points.shape[0]
    homogeneous_points = np.hstack([camera_points, np.ones((num_points, 1))])
    world_homogeneous = (transform_matrix @ homogeneous_points.T).T
    world_points = world_homogeneous[:, :3] / world_homogeneous[:, 3:4]
    return world_points

def transform_quaternions(camera_quaternions: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transforms quaternions from camera coordinates to world coordinates by applying the rotation 
    component of the transform matrix.
    
    Parameters:
        camera_quaternions (np.ndarray): Array of shape (N, 4) with quaternions in (qx, qy, qz, qw) format.
        transform_matrix (np.ndarray): 4x4 transformation matrix.
    
    Returns:
        np.ndarray: Array of shape (N, 4) with world quaternions.
    """
    # Extract rotation part of the transform matrix.
    R_cam_to_world = R.from_matrix(transform_matrix[:3, :3])
    camera_rotations = R.from_quat(camera_quaternions)
    # Compose the rotations: world_rotation = R_cam_to_world * camera_rotation
    world_rotations = R_cam_to_world * camera_rotations
    return world_rotations.as_quat()

def main():
    # File paths (update these paths as needed)
    extrinsics_file = "camera_extrinsics.json"
    input_csv = "input_data.csv"
    output_csv = "output_data.csv"
    
    # Load the camera extrinsics and compute the transformation matrix.
    T = load_extrinsics(extrinsics_file)
    print("Transformation Matrix (Camera -> World):\n", T)
    
    # Load the dataset (CSV) with hand and object positions & orientations in camera coordinates.
    df = pd.read_csv(input_csv)
    
    # Extract hand data (positions and quaternions)
    hand_pos = df[['hand_x', 'hand_y', 'hand_z']].values  # shape (N, 3)
    hand_quat = df[['hand_qx', 'hand_qy', 'hand_qz', 'hand_qw']].values  # shape (N, 4)
    
    # Extract object data (positions and quaternions)
    object_pos = df[['object_x', 'object_y', 'object_z']].values  # shape (N, 3)
    object_quat = df[['object_qx', 'object_qy', 'object_qz', 'object_qw']].values  # shape (N, 4)
    
    # Compute world coordinates for hand
    world_hand_pos = transform_points(hand_pos, T)
    world_hand_quat = transform_quaternions(hand_quat, T)
    
    # Compute world coordinates for object
    world_object_pos = transform_points(object_pos, T)
    world_object_quat = transform_quaternions(object_quat, T)
    
    # Append new columns to the DataFrame for the world coordinates.
    # For hand:
    df['hand_x_world'] = world_hand_pos[:, 0]
    df['hand_y_world'] = world_hand_pos[:, 1]
    df['hand_z_world'] = world_hand_pos[:, 2]
    df['hand_qx_world'] = world_hand_quat[:, 0]
    df['hand_qy_world'] = world_hand_quat[:, 1]
    df['hand_qz_world'] = world_hand_quat[:, 2]
    df['hand_qw_world'] = world_hand_quat[:, 3]
    
    # For object:
    df['object_x_world'] = world_object_pos[:, 0]
    df['object_y_world'] = world_object_pos[:, 1]
    df['object_z_world'] = world_object_pos[:, 2]
    df['object_qx_world'] = world_object_quat[:, 0]
    df['object_qy_world'] = world_object_quat[:, 1]
    df['object_qz_world'] = world_object_quat[:, 2]
    df['object_qw_world'] = world_object_quat[:, 3]
    
    # Save the new DataFrame to a CSV file.
    df.to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")

if __name__ == "__main__":
    main()
