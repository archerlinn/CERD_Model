#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
import open3d as o3d

# --- Configuration ---
# File paths (update these with your actual file locations)
HAND_FILE = "/home/archer/cerd_data/pour_water_01/hand/joints_npy/timestamp1740349029/pour_water_01_1740349029.000_21_0_hand_3d.npy"
OBJECT_FILE1 = "/home/archer/cerd_data/pour_water_01/point_clouds/timestamp1740349029/pour_water_01_1740349029.000_21_mask_tilted_bottle.ply"
OBJECT_FILE2 = "/home/archer/cerd_data/pour_water_01/point_clouds/timestamp1740349029/pour_water_01_1740349029.000_21_mask_cup.ply"

# Manual offset (in millimeters) to be applied to the hand keypoints.
MANUAL_OFFSET = np.array([40.0, 50.0, 300.0])  # [X, Y, Z]

# Number of closest object points to use when computing the contact centroid.
NUM_CLOSEST_POINTS = 5

def load_hand_keypoints(file_path):
    """
    Loads hand keypoints from a .npy file.
    Assumes keypoints are in meters and converts them to millimeters.
    Returns an (N,3) array.
    """
    keypoints_m = np.load(file_path)
    return keypoints_m * 1000.0  # convert to mm

def load_object_points(ply_path):
    """
    Loads a point cloud from a .ply file using Open3D and converts to a NumPy array.
    Assumes the point cloud is already in millimeters.
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)

def compute_hand_centroid(hand_points):
    """
    Computes the centroid of the hand points.
    """
    return np.mean(hand_points, axis=0)

def compute_contact_centroid(obj_points, hand_centroid, num_points=NUM_CLOSEST_POINTS):
    """
    Computes the centroid of the 'num_points' closest points in the object to the hand_centroid.
    
    Args:
        obj_points (np.ndarray): Array of shape (M,3) for the object.
        hand_centroid (np.ndarray): Array of shape (3,).
        num_points (int): Number of closest points to use.
    
    Returns:
        contact_centroid (np.ndarray): The centroid of the selected closest points.
    """
    # Compute distances from hand_centroid to all object points.
    dists = np.linalg.norm(obj_points - hand_centroid, axis=1)
    sorted_indices = np.argsort(dists)
    num_available = min(num_points, len(sorted_indices))
    closest_indices = sorted_indices[:num_available]
    contact_centroid = np.mean(obj_points[closest_indices], axis=0)
    return contact_centroid

def visualize_scene(hand_points, obj_points1, obj_points2, hand_centroid, contact_centroid1, contact_centroid2):
    """
    Visualizes the hand keypoints, two object point clouds, and the computed centroids.
    - Hand points: red dots.
    - Object1: blue dots.
    - Object2: green dots.
    - Hand centroid: large red star.
    - Contact centroid for object1: large blue star.
    - Contact centroid for object2: large green star.
    """
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot object 1 points in blue.
    ax.scatter(obj_points1[:,0], obj_points1[:,1], obj_points1[:,2], c='blue', s=1, label='Object 1')
    # Plot object 2 points in green.
    ax.scatter(obj_points2[:,0], obj_points2[:,1], obj_points2[:,2], c='green', s=1, label='Object 2')
    # Plot hand keypoints in red.
    ax.scatter(hand_points[:,0], hand_points[:,1], hand_points[:,2], c='red', s=50, marker='o', label='Hand Points')
    # Plot centroids as stars.
    ax.scatter(hand_centroid[0], hand_centroid[1], hand_centroid[2], c='red', s=200, marker='*', label='Hand Centroid')
    ax.scatter(contact_centroid1[0], contact_centroid1[1], contact_centroid1[2], c='blue', s=200, marker='*', label='Object1 Contact Centroid')
    ax.scatter(contact_centroid2[0], contact_centroid2[1], contact_centroid2[2], c='green', s=200, marker='*', label='Object2 Contact Centroid')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Hand, Objects, and Centroids Visualization')
    ax.legend()
    plt.show()

def main():
    # Load hand keypoints and convert to mm.
    hand_points = load_hand_keypoints(HAND_FILE)
    # Apply manual offset.
    hand_points_offset = hand_points + MANUAL_OFFSET
    # Compute hand centroid.
    hand_centroid = compute_hand_centroid(hand_points_offset)
    
    # Load object point clouds.
    obj_points1 = load_object_points(OBJECT_FILE1)
    obj_points2 = load_object_points(OBJECT_FILE2)
    
    # Compute contact centroids for each object using the 50 closest points to the hand centroid.
    contact_centroid1 = compute_contact_centroid(obj_points1, hand_centroid, NUM_CLOSEST_POINTS)
    contact_centroid2 = compute_contact_centroid(obj_points2, hand_centroid, NUM_CLOSEST_POINTS)
    
    # Debug printouts.
    print("Hand centroid (offset):", hand_centroid)
    print("Object 1 contact centroid (50 closest):", contact_centroid1)
    print("Object 2 contact centroid (50 closest):", contact_centroid2)
    
    # Visualize the scene.
    visualize_scene(hand_points_offset, obj_points1, obj_points2, hand_centroid, contact_centroid1, contact_centroid2)

if __name__ == "__main__":
    main()
