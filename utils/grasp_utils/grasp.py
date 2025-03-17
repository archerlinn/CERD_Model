#!/usr/bin/env python
"""
compare_grasp_state_closest50.py

This script loads 3D hand keypoints (saved as .npy files from HaMeR) and object point clouds
(saved as .ply files from Grounded-SAM2), applies an automatic offset in the z dimension to align
the hand keypoints with the object's depth, and then computes a grasp detection metric as follows:
For each matched hand–object pair (matched by sequence, timestamp, and frame):
  1. It computes d_thumb: the distance from the thumb’s second joint (index 3) to the contact centroid,
     where the contact centroid is the average of the NUM_CLOSEST_POINTS closest object points to the thumb joint.
  2. For each candidate finger (using their third joint at indices 7, 11, 15, and 19), it computes the contact centroid
     based on the NUM_CLOSEST_POINTS closest object points to that candidate, and then computes d_candidate.
  3. If for any candidate the sum (d_thumb + d_candidate) is below SUM_THRESH, the grasp is detected (grasp = 1);
     otherwise, grasp = 0.
Left-hand files (hand_side == "1") are dropped.
Results are saved (sorted by timestamp and frame) to a CSV file.

Joint 0: Wrist
Joints 1~4: Thumb (1 = base, 2 = first joint, 3 = second joint, 4 = tip)
Joints 5~8: Index Finger (5 = base, 6 = first joint, 7 = second joint, 8 = tip)
Joints 9~12: Middle Finger (9 = base, 10 = first joint, 11 = second joint, 12 = tip)
Joints 13~16: Ring Finger (13 = base, 14 = first joint, 15 = second joint, 16 = tip)
Joints 17~20: Pinky Finger (17 = base, 18 = first joint, 19 = second joint, 20 = tip)
"""
import os
import csv
import glob
import numpy as np
import open3d as o3d

# === Configuration Parameters ===
HAND_KEYPOINTS_FOLDER = "/home/archer/cerd_data/pour_water_07/hand/joints_npy"
OBJECT_PLY_FOLDER = "/home/archer/cerd_data/pour_water_07/point_clouds"
OUTPUT_CSV = "/home/archer/cerd_data/pour_water_07/grasp.csv"

SUM_THRESH = 100      # Sum threshold (in mm) for (d_thumb + d_candidate)
NUM_CLOSEST_POINTS = 20  # Number of closest object points to compute each contact centroid

# === Helper Functions for File ID Parsing ===
def parse_hand_file_id(hand_id: str) -> tuple:
    """
    Expected hand filename format:
      "pour_water_01_1740349022.000_28_0_hand_3d"
    Returns:
      (sequence, timestamp, frame, hand_side)
    """
    parts = hand_id.split("_")
    if len(parts) >= 8:
        sequence = "_".join(parts[:3])
        timestamp = parts[3]
        frame = parts[4]
        hand_side = parts[5]  # "0" for right hand, "1" for left hand
        return sequence, timestamp, frame, hand_side
    return "", "", "", ""

def parse_object_file_id(obj_id: str) -> tuple:
    """
    Expected object filename format:
      "pour_water_01_1740349022.000_28_mask_cup"
    Returns:
      (sequence, timestamp, frame)
    """
    parts = obj_id.split("_")
    if len(parts) >= 5:
        sequence = "_".join(parts[:3])
        timestamp = parts[3]
        frame = parts[4]
        return sequence, timestamp, frame
    return "", "", ""

def parse_timestamp_to_float(ts_str: str) -> float:
    try:
        return float(ts_str)
    except ValueError:
        return 0.0

# === Data Loading Functions ===
def load_hand_keypoints(folder):
    """
    Recursively load .npy hand keypoint files (assumed in meters) and convert them to mm.
    Returns a dictionary mapping file ID to a 21x3 array.
    """
    hand_files = glob.glob(os.path.join(folder, "**", "*.npy"), recursive=True)
    hand_data = {}
    for f in hand_files:
        hand_id = os.path.splitext(os.path.basename(f))[0]
        keypoints_m = np.load(f)
        keypoints_mm = keypoints_m * 1000.0
        hand_data[hand_id] = keypoints_mm
    return hand_data

def load_object_point_clouds(folder):
    """
    Recursively load .ply object point cloud files (assumed in mm).
    Returns a dictionary mapping file ID to an Open3D point cloud.
    """
    ply_files = glob.glob(os.path.join(folder, "**", "*.ply"), recursive=True)
    obj_data = {}
    for f in ply_files:
        obj_id = os.path.splitext(os.path.basename(f))[0]
        pcd = o3d.io.read_point_cloud(f)
        obj_data[obj_id] = pcd
    return obj_data

# === Contact Centroid Computation ===
def compute_contact_centroid(obj_points, joint, num_points):
    """
    Given an array of object points and a joint position, find the num_points closest points,
    compute their mean (the contact centroid), and return the centroid and the distance from joint.
    """
    dists = np.linalg.norm(obj_points - joint, axis=1)
    sorted_indices = np.argsort(dists)
    n = min(num_points, len(sorted_indices))
    closest = obj_points[sorted_indices[:n]]
    centroid = np.mean(closest, axis=0)
    distance = np.linalg.norm(joint - centroid)
    return centroid, distance

# === Grasp State Computation ===
def compute_grasp_state_by_sum(hand_keypoints_mm, obj_pcd, sum_thresh, num_points):
    """
    For each hand–object pair:
      - For the thumb: use its second joint (index 3).
        Compute d_thumb using the num_points closest object points.
      - For each candidate finger: use its third joint (indices 7, 11, 15, 19).
        Compute d_candidate for each candidate.
      - If any candidate yields (d_thumb + d_candidate) < sum_thresh, return grasp = 1.
    Returns:
      grasp_state (int), min_total_distance (float), thumb_joint, thumb_contact_centroid.
    """
    obj_points = np.asarray(obj_pcd.points)
    if obj_points.size == 0:
        return 0, None, hand_keypoints_mm[3], None

    # Thumb's second joint (index 3)
    thumb_joint = hand_keypoints_mm[3]
    thumb_centroid, d_thumb = compute_contact_centroid(obj_points, thumb_joint, num_points)

    candidate_indices = [7, 11, 15, 19]
    totals = []
    for idx in candidate_indices:
        if idx < len(hand_keypoints_mm):
            candidate_joint = hand_keypoints_mm[idx]
            candidate_centroid, d_candidate = compute_contact_centroid(obj_points, candidate_joint, num_points)
            totals.append(d_thumb + d_candidate)
    if totals:
        min_total = min(totals)
    else:
        min_total = None

    grasp_state = 1 if (min_total is not None and min_total < sum_thresh) else 0
    return grasp_state, min_total, thumb_joint, thumb_centroid

# === Main Processing ===
def main():
    hand_data = load_hand_keypoints(HAND_KEYPOINTS_FOLDER)
    obj_data = load_object_point_clouds(OBJECT_PLY_FOLDER)

    results = []  # Each result: (timestamp, frame, hand_id, obj_id, grasp_state, total_distance)

    for hand_id, hand_kps_mm in hand_data.items():
        hand_seq, hand_ts_str, hand_frame, hand_side = parse_hand_file_id(hand_id)
        # Drop left-hand files.
        if hand_side == "1":
            continue

        ts_val = parse_timestamp_to_float(hand_ts_str)
        # Compute auto offset in z using the thumb's second joint (index 3)
        # This aligns the hand keypoints with the object's depth.
        matched = False
        for obj_id, obj_pcd in obj_data.items():
            obj_seq, obj_ts_str, obj_frame = parse_object_file_id(obj_id)
            if hand_seq == obj_seq and hand_ts_str == obj_ts_str and hand_frame == obj_frame:
                obj_pts = np.asarray(obj_pcd.points)
                if obj_pts.size > 0:
                    auto_offset_z = np.mean(obj_pts[:, 2]) - hand_kps_mm[3, 2]
                else:
                    auto_offset_z = 0
                hand_kps_mm_corrected = hand_kps_mm.copy()
                hand_kps_mm_corrected[:, 2] += auto_offset_z

                grasp_state, total_distance, thumb_joint, thumb_centroid = compute_grasp_state_by_sum(
                    hand_kps_mm_corrected, obj_pcd, SUM_THRESH, NUM_CLOSEST_POINTS
                )
                results.append((ts_val, int(hand_frame), hand_id, obj_id, grasp_state, total_distance))
                matched = True
        if not matched:
            print(f"No object matched for hand {hand_id} (seq={hand_seq}, timestamp={hand_ts_str}, frame={hand_frame})")
    
    results.sort(key=lambda r: (r[0], r[1]))
    
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["hand_id", "object_id", "grasp", "total_distance_mm"])
        for res in results:
            ts, frame, hand_id, obj_id, grasp_state, total_distance = res
            writer.writerow([hand_id, obj_id, grasp_state, f"{total_distance:.4f}" if total_distance is not None else "None"])
            print(f"Time {ts} | Hand {hand_id} vs Object {obj_id}: grasp={grasp_state}, total_distance={total_distance:.2f} mm")
    
    print(f"\nGrasp state results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()