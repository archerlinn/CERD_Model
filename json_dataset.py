import pandas as pd
import json
import numpy as np
import open3d as o3d
from pathlib import Path

def load_keyframe_csv(csv_path):
    """ Load keyframe CSV and structure it into the required format. """
    df = pd.read_csv(csv_path)
    
    # Group by sequence (demonstration)
    demonstrations = {}
    
    for _, row in df.iterrows():
        sequence = row['sequence']
        frame_id = int(row['frame_id'])

        # Ensure demonstration exists
        if sequence not in demonstrations:
            demonstrations[sequence] = {
                "obs": [],
                "actions": {}
            }
        
        # Add observation placeholder (to be filled with point cloud later)
        demonstrations[sequence]["obs"].append([])  

        # Add actions for each keyframe
        demonstrations[sequence]["actions"][frame_id] = {
            "0": row["hand_x"],
            "1": row["hand_y"],
            "2": row["hand_z"],
            "3": row["hand_qx"],
            "4": row["hand_qy"],
            "5": row["hand_qz"],
            "6": int(row["grasp_state"])  # Convert grasp state to int (0 or 1)
        }

    return demonstrations


def load_ply_file(ply_path):
    """ Load a .ply point cloud file and return the point coordinates as a list of lists. """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points).tolist()
    return points


def integrate_observations(demonstrations, ply_files):
    """ Integrate 3D object point clouds into each demonstration. """
    for sequence in demonstrations:
        # Combine both object point clouds into one observation set
        combined_pcd = []
        for ply_path in ply_files:
            combined_pcd.extend(load_ply_file(ply_path))
        
        # Assign the same observation to all frames in the sequence
        demonstrations[sequence]["obs"] = [combined_pcd] * len(demonstrations[sequence]["obs"])
    
    return demonstrations


def save_json(data, output_path):
    """ Save the structured data into a JSON file. """
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def main(keyframe_csv, ply1_path, ply2_path, output_json):
    """ Main function to convert dataset into the required JSON format. """
    # Step 1: Load keyframe data
    demonstrations = load_keyframe_csv(keyframe_csv)

    # Step 2: Integrate point cloud data from .ply files
    ply_files = [ply1_path, ply2_path]
    demonstrations = integrate_observations(demonstrations, ply_files)

    # Step 3: Save JSON
    save_json(demonstrations, output_json)
    print(f"JSON file saved at: {output_json}")


# Example usage:
if __name__ == "__main__":
    keyframe_csv_path = "path/to/keyframe.csv"
    ply1_path = "path/to/object1.ply"
    ply2_path = "path/to/object2.ply"
    output_json_path = "path/to/output.json"

    main(keyframe_csv_path, ply1_path, ply2_path, output_json_path)
