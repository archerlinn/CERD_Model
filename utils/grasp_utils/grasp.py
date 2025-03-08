#!/usr/bin/env python
"""
compare_grasp_state_closest50.py

This script loads 3D hand keypoints (saved as .npy files from HaMeR) and object point clouds
(saved as .ply files from Grounded-SAM2), applies a manual offset to the hand keypoints,
computes the centroid of the hand (from its 21 keypoints), then finds the 50 closest points
in the object point cloud to the hand centroid, computes the centroid of those 50 points,
and finally computes the Euclidean distance between these two centroids.
If that distance is below DIST_THRESH, the grasp is considered closed (grasp_state = 0);
otherwise, it is open (grasp_state = 1).
The results are saved (sorted by timestamp) to a CSV file.

File naming assumptions:
- Hand file names: "pour_water_02_1740349023.000_0_hand_3d.npy" (timestamp is the 4th element).
- Object file names: "1740349023.000_1740349023.000_mask_0.ply" (timestamp is the 1st element).
"""

import os
import csv
import glob
import numpy as np
import open3d as o3d

# --- Configuration Parameters ---
HAND_KEYPOINTS_FOLDER = "/home/archer/code/CERD_Model/dataset/pour_water_02/hand/joints_npy"
OBJECT_PLY_FOLDER = "/home/archer/code/CERD_Model/dataset/pour_water_02/point_clouds"
OUTPUT_CSV = "/home/archer/code/CERD_Model/dataset/pour_water_02/grasp.csv"

# Distance threshold (in mm) for deciding a "grasp" based on the distance between centroids.
DIST_THRESH = 65

# Number of closest object points to use when computing the object's contact centroid.
NUM_CLOSEST_POINTS = 200

# --- Manual Offset Settings (in millimeters) ---
# Adjust these values as needed to shift the hand's position.
MANUAL_OFFSET = np.array([40.0, 50.0, 300.0])

# --- Helper Functions ---

def parse_hand_timestamp(hand_id: str) -> str:
    parts = hand_id.split("_")
    return parts[3] if len(parts) >= 4 else ""

def parse_object_timestamp(obj_id: str) -> str:
    parts = obj_id.split("_")
    return parts[0] if len(parts) >= 1 else ""

def parse_timestamp_to_float(ts_str: str) -> float:
    try:
        return float(ts_str)
    except ValueError:
        return 0.0

def load_hand_keypoints(folder):
    """
    Loads .npy files and returns a dictionary mapping file IDs to a 21x3 array (in mm).
    Assumes hand keypoints are stored in meters.
    """
    hand_files = glob.glob(os.path.join(folder, "*.npy"))
    hand_data = {}
    for f in hand_files:
        base = os.path.basename(f)
        hand_id = os.path.splitext(base)[0]  # e.g. "pour_water_02_1740349023.000_0_hand_3d"
        keypoints_m = np.load(f)            # shape (21, 3) in meters
        keypoints_mm = keypoints_m * 1000.0   # convert to mm
        hand_data[hand_id] = keypoints_mm
    return hand_data

def load_object_point_clouds(folder):
    """
    Loads .ply files and returns a dictionary mapping file IDs to an Open3D point cloud.
    Assumes point clouds are in millimeters.
    """
    ply_files = glob.glob(os.path.join(folder, "*.ply"))
    obj_data = {}
    for f in ply_files:
        base = os.path.basename(f)
        obj_id = os.path.splitext(base)[0]
        pcd = o3d.io.read_point_cloud(f)
        obj_data[obj_id] = pcd
    return obj_data

def compute_grasp_state_closest50(hand_keypoints_mm, obj_pcd, dist_thresh, num_points=NUM_CLOSEST_POINTS):
    """
    Computes the grasp state as follows:
      1. Compute the hand centroid (mean of the 21 keypoints).
      2. Compute the Euclidean distances from the hand centroid to each object point.
      3. Select the 'num_points' closest object points.
      4. Compute the centroid of these selected object points (the "contact centroid").
      5. Compute the Euclidean distance between the hand centroid and the contact centroid.
      6. If that distance is less than dist_thresh, return grasp_state = 0 (closed); else 1.
      
    Returns:
        grasp_state (int), distance (float), hand_centroid (np.ndarray), contact_centroid (np.ndarray)
    """
    # Compute hand centroid.
    hand_centroid = np.mean(hand_keypoints_mm, axis=0)
    
    obj_points = np.asarray(obj_pcd.points)
    if obj_points.size == 0:
        return 1, None, hand_centroid, None

    # Compute distances from hand centroid to each object point.
    dists = np.linalg.norm(obj_points - hand_centroid, axis=1)
    # Get indices of the closest 'num_points' object points.
    sorted_indices = np.argsort(dists)
    num_available = min(num_points, len(sorted_indices))
    closest_indices = sorted_indices[:num_available]
    # Compute the contact centroid from these points.
    contact_centroid = np.median(obj_points[closest_indices], axis=0)
    # Compute the Euclidean distance between hand centroid and contact centroid.
    contact_distance = np.linalg.norm(hand_centroid - contact_centroid)
    grasp_state = 1 if contact_distance < dist_thresh else 0
    return grasp_state, contact_distance, hand_centroid, contact_centroid

def main():
    # 1. Load data.
    hand_data = load_hand_keypoints(HAND_KEYPOINTS_FOLDER)  # dict: hand_id -> 21x3 (mm)
    obj_data = load_object_point_clouds(OBJECT_PLY_FOLDER)   # dict: obj_id -> pointcloud (mm)

    results = []  # each element: (timestamp, hand_id, obj_id, grasp_state, contact_distance)

    # 2. For each hand file, apply the manual offset and compare with matching object files.
    for hand_id, hand_kps_mm in hand_data.items():
        hand_ts_str = parse_hand_timestamp(hand_id)
        hand_ts_val = parse_timestamp_to_float(hand_ts_str)
        
        # Apply the manual offset.
        hand_kps_mm_corrected = hand_kps_mm + MANUAL_OFFSET
        
        matched_any = False
        for obj_id, obj_pcd in obj_data.items():
            obj_ts_str = parse_object_timestamp(obj_id)
            if hand_ts_str and (hand_ts_str == obj_ts_str):
                grasp_state, contact_distance, hand_centroid, contact_centroid = compute_grasp_state_closest50(hand_kps_mm_corrected, obj_pcd, DIST_THRESH)
                results.append((hand_ts_val, hand_id, obj_id, grasp_state, contact_distance))
                matched_any = True
        if not matched_any:
            print(f"No object matched for hand {hand_id} (timestamp={hand_ts_str})")
    
    # 3. Sort results by ascending timestamp.
    results.sort(key=lambda row: row[0])
    
    # 4. Write results to CSV and print them.
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["hand_id", "object_id", "grasp", "contact_centroid_distance_mm"])
        for row in results:
            ts, hand_id, obj_id, grasp_state, contact_distance = row
            writer.writerow([hand_id, obj_id, grasp_state, f"{contact_distance:.4f}" if contact_distance is not None else "None"])
            print(f"Time {ts} | Hand {hand_id} vs Object {obj_id}: grasp={grasp_state}, contact_centroid_distance={contact_distance:.2f} mm")
    
    print(f"\nGrasp state results saved (sorted) to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
