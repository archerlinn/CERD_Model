#!/usr/bin/env python
"""
merge_and_transform_recursive.py

This script merges hand joints CSV files, hand orientation CSV files, and object PLY files
from three root folders (which may contain subfolders based on timestamp). It assumes that
corresponding files share the same base filename (ignoring suffixes).

Expected file naming convention example:
  Hand joints file:       "pour_water_01_1740349022.000_28_0_joints.csv"
  Hand orientation file:  "pour_water_01_1740349022.000_28_0_orientation.csv"
  Object PLY file:        "pour_water_01_1740349022.000_28_mask_tilted_bottle.ply"

Files corresponding to left-hand (hand_side == "1") are dropped.
The script loads camera extrinsics from a JSON file to compute a 4×4 transformation matrix,
transforms the hand position (from the wrist, joint_index == 0) and the object position
(computed as the centroid of all object points) from camera to world coordinates, and writes the
final merged CSV. The output is sorted by timestamp and frame.
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# === Configuration ===
extrinsics_file = "/home/archer/cerd_data/calibration.json"     # Path to extrinsics JSON.
hand_joints_root = "/home/archer/cerd_data/pour_water_01/hand/joints_csv"   # Root folder for hand joints CSV files.
hand_orient_root = "/home/archer/cerd_data/pour_water_01/hand/orientation_csv" # Root folder for hand orientation CSV files.
object_ply_root = "/home/archer/cerd_data/pour_water_01/point_clouds"         # Root folder for object PLY files.
output_csv = "/home/archer/cerd_data/pour_water_01/world_pos.csv"             # Output CSV file.

# Suffixes to remove from base filenames:
HAND_JOINTS_SUFFIX = "_joints"      # Remove only "_joints" (hand side remains)
HAND_ORIENT_SUFFIX = "_orientation" # e.g., file name ends with "_orientation.csv"
OBJECT_SUFFIX = "_mask"             # Object files include "_mask" somewhere in their names

# === Extrinsics and Transformation Functions ===
def load_extrinsics(json_file: str) -> np.ndarray:
    with open(json_file, 'r') as f:
        extrinsics = json.load(f)
    rx = extrinsics["extrinsic_matrix"]["rotation_offsets"]["RX"]
    cv = extrinsics["extrinsic_matrix"]["rotation_offsets"]["CV"]
    rz = extrinsics["extrinsic_matrix"]["rotation_offsets"]["RZ"]
    angles = [rx, cv, rz]
    rotation = R.from_euler('xyz', angles)
    tx, ty = extrinsics["extrinsic_matrix"]["translation_vector"]
    baseline_mm = extrinsics["extrinsic_matrix"]["baseline"]
    tz = baseline_mm / 1000.0  # convert mm to m
    translation = np.array([tx, ty, tz])
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = translation
    return T

def transform_points(camera_points: np.ndarray, T: np.ndarray) -> np.ndarray:
    num_points = camera_points.shape[0]
    homo = np.hstack([camera_points, np.ones((num_points, 1))])
    world_homo = (T @ homo.T).T
    return world_homo[:, :3] / world_homo[:, 3:4]

def transform_quaternions(camera_quaternions: np.ndarray, T: np.ndarray) -> np.ndarray:
    R_cam_to_world = R.from_matrix(T[:3, :3])
    cam_rots = R.from_quat(camera_quaternions)
    world_rots = R_cam_to_world * cam_rots
    return world_rots.as_quat()

# === Data Loading Functions ===
def load_hand_joints(csv_file: str) -> np.ndarray:
    df = pd.read_csv(csv_file)
    df["joint_index"] = df["joint_index"].astype(int)
    df = df.sort_values("joint_index")
    return df[["x", "y", "z"]].values

def load_hand_orientation(csv_file: str) -> np.ndarray:
    df = pd.read_csv(csv_file)
    angles = df.iloc[0][["ox", "oy", "oz"]].values.astype(float)
    quat = R.from_euler('xyz', angles, degrees=True).as_quat()
    return quat

def compute_object_centroid(ply_file: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_file)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return np.array([0, 0, 0])
    return np.mean(pts, axis=0)

# === Recursive File Collection ===
def get_files_recursive(root: str, pattern: str, suffix_to_remove: str, use_split: bool = False) -> dict:
    files = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    file_dict = {}
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        if use_split and suffix_to_remove in base:
            # Split at the first occurrence of the suffix
            key = base.split(suffix_to_remove)[0]
            # Check if the key already has a hand-side (we expect 6 parts when split by '_')
            parts = key.split("_")
            if len(parts) == 5:
                # Append default hand-side ("0") to match hand file keys
                key = key + "_0"
        elif base.endswith(suffix_to_remove):
            key = base.replace(suffix_to_remove, "")
        else:
            key = base
        file_dict[key] = f
    return file_dict

# === Merging Files ===
def merge_files(hand_joints_dict: dict, hand_orient_dict: dict, object_dict: dict) -> pd.DataFrame:
    common_keys = set(hand_joints_dict.keys()) & set(hand_orient_dict.keys()) & set(object_dict.keys())
    # Debug prints:
    print("Hand joints keys:", list(hand_joints_dict.keys()))
    print("Hand orientation keys:", list(hand_orient_dict.keys()))
    print("Object keys:", list(object_dict.keys()))
    
    merged_rows = []
    for key in common_keys:
        seq, ts, frame, hand_side = parse_hand_file_id_from_filename(key)
        # Drop left-hand files (hand_side == "1")
        if hand_side == "1":
            continue
        hand_joints = load_hand_joints(hand_joints_dict[key])
        hand_orient = load_hand_orientation(hand_orient_dict[key])
        obj_centroid = compute_object_centroid(object_dict[key])
        row = {
            "file_id": key,
            "sequence": seq,
            "timestamp": ts,
            "frame": frame,
            "hand_x": hand_joints[0, 0],
            "hand_y": hand_joints[0, 1],
            "hand_z": hand_joints[0, 2],
            "hand_qx": hand_orient[0],
            "hand_qy": hand_orient[1],
            "hand_qz": hand_orient[2],
            "hand_qw": hand_orient[3],
            "object_x": obj_centroid[0],
            "object_y": obj_centroid[1],
            "object_z": obj_centroid[2],
            "object_qx": 0,
            "object_qy": 0,
            "object_qz": 0,
            "object_qw": 1
        }
        merged_rows.append(row)
    if merged_rows:
        df = pd.DataFrame(merged_rows)
        df["timestamp_val"] = df["timestamp"].apply(parse_timestamp_to_float)
        df["frame_val"] = df["frame"].astype(int)
        df.sort_values(["timestamp_val", "frame_val"], inplace=True)
        return df
    else:
        return pd.DataFrame()

def parse_hand_file_id_from_filename(filename: str) -> tuple:
    parts = filename.split("_")
    # Expecting keys like: pour_water_01_1740349023.000_14_0
    if len(parts) >= 6:
        sequence = "_".join(parts[:3])
        timestamp = parts[3]
        frame = parts[4]
        hand_side = parts[5]
        return sequence, timestamp, frame, hand_side
    # If there are only 5 parts, assume default right-hand ("0")
    if len(parts) == 5:
        sequence = "_".join(parts[:3])
        timestamp = parts[3]
        frame = parts[4]
        hand_side = "0"
        return sequence, timestamp, frame, hand_side
    return "", "", "", ""

def parse_object_file_id_from_filename(filename: str) -> tuple:
    parts = filename.split("_")
    if len(parts) >= 5:
        sequence = "_".join(parts[:3])
        timestamp = parts[3]
        frame = parts[4]
        return sequence, timestamp, frame
    return "", "", ""

def parse_timestamp_to_float(ts: str) -> float:
    try:
        return float(ts)
    except ValueError:
        return 0.0

# === Main Script ===
def main():
    hand_joints_files = get_files_recursive(hand_joints_root, "*.csv", HAND_JOINTS_SUFFIX)
    hand_orient_files = get_files_recursive(hand_orient_root, "*.csv", HAND_ORIENT_SUFFIX)
    # For object files, use splitting so that any filename containing "_mask" is split properly.
    object_files = get_files_recursive(object_ply_root, "*.ply", OBJECT_SUFFIX, use_split=True)
    
    df_cam = merge_files(hand_joints_files, hand_orient_files, object_files)
    if df_cam.empty:
        print("No common files found. Exiting.")
        return
    print(f"Merged {len(df_cam)} entries from hand joints, hand orientation, and object PLY files.")
    
    T = load_extrinsics(extrinsics_file)
    print("Transformation Matrix (Camera -> World):")
    print(T)
    
    hand_pos = df_cam[["hand_x", "hand_y", "hand_z"]].values
    hand_quat = df_cam[["hand_qx", "hand_qy", "hand_qz", "hand_qw"]].values
    world_hand_pos = transform_points(hand_pos, T)
    world_hand_quat = transform_quaternions(hand_quat, T)
    
    object_pos = df_cam[["object_x", "object_y", "object_z"]].values
    object_quat = df_cam[["object_qx", "object_qy", "object_qz", "object_qw"]].values
    world_object_pos = transform_points(object_pos, T)
    world_object_quat = transform_quaternions(object_quat, T)
    
    df_cam["hand_x_world"] = world_hand_pos[:, 0]
    df_cam["hand_y_world"] = world_hand_pos[:, 1]
    df_cam["hand_z_world"] = world_hand_pos[:, 2]
    df_cam["hand_qx_world"] = world_hand_quat[:, 0]
    df_cam["hand_qy_world"] = world_hand_quat[:, 1]
    df_cam["hand_qz_world"] = world_hand_quat[:, 2]
    df_cam["hand_qw_world"] = world_hand_quat[:, 3]
    
    df_cam["object_x_world"] = world_object_pos[:, 0]
    df_cam["object_y_world"] = world_object_pos[:, 1]
    df_cam["object_z_world"] = world_object_pos[:, 2]
    df_cam["object_qx_world"] = world_object_quat[:, 0]
    df_cam["object_qy_world"] = world_object_quat[:, 1]
    df_cam["object_qz_world"] = world_object_quat[:, 2]
    df_cam["object_qw_world"] = world_object_quat[:, 3]
    
    df_cam.drop(columns=["timestamp_val", "frame_val"], inplace=True)
    
    df_cam.to_csv(output_csv, index=False)
    print(f"Output CSV saved to {output_csv}")

if __name__ == "__main__":
    main()
