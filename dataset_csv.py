#!/usr/bin/env python3
import os
import re
import csv
from plyfile import PlyData

# Specify the root directories and output CSV path here:
joints_root = "/home/archer/cerd_data/pour_water_07/hand/joints_csv"          # Folder containing hand joints CSV files
orientation_root = "/home/archer/cerd_data/pour_water_07/hand/orientation_csv"  # Folder containing hand orientation CSV files
ply_root = "/home/archer/cerd_data/pour_water_07/point_clouds"                  # Folder containing object PLY files
grasp_csv_path = "/home/archer/cerd_data/pour_water_07/grasp.csv"   # Grasp state CSV file
output_csv = "/home/archer/cerd_data/pour_water_07/data.csv"          # Output CSV file

# ------------- FILENAME PARSERS -------------
def parse_joints_filename(filename):
    """
    Expected joints filename format:
      pour_water_01_1740349022.000_22_0_joints.csv
    Interprets:
      sequence: "pour_water_01"
      timestamp: "1740349022.000"
      frame: "22"
      hand: "0" (0 = right hand, 1 = left hand)
    """
    pattern = re.compile(
        r'^(?P<sequence>.+?)_(?P<timestamp>\d+\.\d+?)_(?P<frame>\d+?)_(?P<hand>[01])_joints\.csv$'
    )
    m = pattern.match(filename)
    return m.groupdict() if m else None

def parse_orientation_filename(filename):
    """
    Expected orientation filename format:
      pour_water_01_1740349022.000_22_0_orientation.csv
    """
    pattern = re.compile(
        r'^(?P<sequence>.+?)_(?P<timestamp>\d+\.\d+?)_(?P<frame>\d+?)_(?P<hand>[01])_orientation\.csv$'
    )
    m = pattern.match(filename)
    return m.groupdict() if m else None

def parse_ply_filename(filename):
    """
    Expected PLY filename format:
      pour_water_01_1740349022.000_22_mask_tilted_bottle.ply
    Interprets:
      sequence: "pour_water_01"
      timestamp: "1740349022.000"
      frame: "22"
      object: "tilted_bottle" (from the part after "mask_")
    """
    pattern = re.compile(
        r'^(?P<sequence>.+?)_(?P<timestamp>\d+\.\d+?)_(?P<frame>\d+?)_mask_(?P<object>.+)\.ply$'
    )
    m = pattern.match(filename)
    return m.groupdict() if m else None

# ------------- FILE HANDLING FUNCTIONS -------------
def find_files(root_dir, file_extension):
    """ Recursively find files with the given extension under root_dir. """
    return [
        os.path.join(dirpath, fname)
        for dirpath, _, filenames in os.walk(root_dir)
        for fname in filenames if fname.endswith(file_extension)
    ]

def compute_hand_centroid(filepath):
    """ Computes centroid (average) of all joints from a joints CSV file. """
    total_x, total_y, total_z, count = 0.0, 0.0, 0.0, 0
    try:
        with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
                total_x += x
                total_y += y
                total_z += z
                count += 1
    except Exception as e:
        print(f"Error reading joints file {filepath}: {e}")
    if count == 0:
        return {}
    return {
        "hand_x": str(total_x / count),
        "hand_y": str(total_y / count),
        "hand_z": str(total_z / count)
    }

def read_orientation_file(filepath):
    """ Reads hand orientation (ox, oy, oz) from an orientation CSV file. """
    try:
        with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader, {})
            return {
                "hand_qx": row.get("ox", "").strip(),
                "hand_qy": row.get("oy", "").strip(),
                "hand_qz": row.get("oz", "").strip(),
                "hand_qw": "1"
            }
    except Exception as e:
        print(f"Error reading orientation file {filepath}: {e}")
        return {}

def compute_ply_centroid(filepath):
    """ Uses PlyData to compute the centroid of a PLY object's vertices. """
    try:
        plydata = PlyData.read(filepath)
        vertices = plydata['vertex']
        x_vals, y_vals, z_vals = vertices['x'], vertices['y'], vertices['z']
        return {
            "object_x": str(sum(x_vals) / len(x_vals)),
            "object_y": str(sum(y_vals) / len(y_vals)),
            "object_z": str(sum(z_vals) / len(z_vals))
        }
    except Exception as e:
        print(f"Error reading PLY file {filepath}: {e}")
        return {}

def load_grasp_csv(filepath):
    """ Loads the grasp state CSV into a dictionary for quick lookup. """
    grasp_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hand_id = row["hand_id"].strip()
                object_id = row["object_id"].strip()
                grasp = row["grasp"].strip()
                total_distance = row["total_distance_mm"].strip()
                grasp_dict[(hand_id, object_id)] = (grasp, total_distance)
    except Exception as e:
        print(f"Error loading grasp CSV {filepath}: {e}")
    return grasp_dict

# ------------- MAIN MERGE LOGIC -------------
def main():
    # Load grasp states for fast lookup.
    grasp_lookup = load_grasp_csv(grasp_csv_path)

    # Build dictionaries for joints and orientation files.
    # Keys: (sequence, timestamp, frame, hand)
    joints_files = {}
    for f in find_files(joints_root, ".csv"):
        info = parse_joints_filename(os.path.basename(f))
        if info:
            key = (info['sequence'], info['timestamp'], info['frame'], info['hand'])
            joints_files[key] = f

    orientation_files = {}
    for f in find_files(orientation_root, ".csv"):
        info = parse_orientation_filename(os.path.basename(f))
        if info:
            key = (info['sequence'], info['timestamp'], info['frame'], info['hand'])
            orientation_files[key] = f

    # Group PLY files by (sequence, timestamp, frame)
    ply_files = {}
    for f in find_files(ply_root, ".ply"):
        info = parse_ply_filename(os.path.basename(f))
        if info:
            key = (info['sequence'], info['timestamp'], info['frame'])
            ply_files.setdefault(key, []).append((f, info.get("object", "").strip()))

    merged_rows = []
    # We will assign frame_id later as group number per unique (sequence, timestamp, frame)
    for key, joints_path in joints_files.items():
        sequence, timestamp, frame, hand = key
        ply_key = (sequence, timestamp, frame)
        if key not in orientation_files:
            print(f"Missing orientation file for key: {key}")
            continue
        if ply_key not in ply_files:
            print(f"Missing PLY files for key: {ply_key}")
            continue

        hand_centroid = compute_hand_centroid(joints_path)
        if not hand_centroid:
            print(f"Could not compute hand centroid for {joints_path}")
            continue

        orient_data = read_orientation_file(orientation_files[key])
        # Construct hand_id as in grasp CSV: e.g., "pour_water_01_1740349022.000_22_0_hand_3d"
        hand_id = f"{sequence}_{timestamp}_{frame}_{hand}_hand_3d"

        for ply_path, object_name in ply_files[ply_key]:
            object_centroid = compute_ply_centroid(ply_path)
            if not object_centroid:
                print(f"Could not compute object centroid for {ply_path}")
                continue

            # Construct object_id as in grasp CSV: e.g., "pour_water_01_1740349022.000_22_mask_tilted_bottle"
            object_id = f"{sequence}_{timestamp}_{frame}_mask_{object_name}"
            grasp, total_distance = grasp_lookup.get((hand_id, object_id), ("", ""))

            merged_rows.append({
                "frame_id": None,  # will assign later
                "sequence": sequence,
                "timestamp": timestamp,
                "frame": frame,
                "hand_x": hand_centroid.get("hand_x", ""),
                "hand_y": hand_centroid.get("hand_y", ""),
                "hand_z": hand_centroid.get("hand_z", ""),
                "hand_qx": orient_data.get("hand_qx", ""),
                "hand_qy": orient_data.get("hand_qy", ""),
                "hand_qz": orient_data.get("hand_qz", ""),
                "hand_qw": orient_data.get("hand_qw", ""),
                "object_x": object_centroid.get("object_x", ""),
                "object_y": object_centroid.get("object_y", ""),
                "object_z": object_centroid.get("object_z", ""),
                "grasp_state": grasp,  # renamed column
                "total_distance_mm": total_distance
            })

    # --- SORT THE ROWS BY TIMESTAMP AND FRAME ---
    try:
        merged_rows.sort(key=lambda row: (float(row["timestamp"]), int(row["frame"])))
    except Exception as e:
        print(f"Error sorting rows: {e}")

    # --- ASSIGN FILE ID AS A GROUP NUMBER FOR EACH UNIQUE (sequence, timestamp, frame) ---
    group_id = 0
    last_group = None
    for row in merged_rows:
        current_group = (row["sequence"], row["timestamp"], row["frame"])
        if current_group != last_group:
            group_id += 1
            last_group = current_group
        row["frame_id"] = group_id

    # Write output CSV.
    fieldnames = [
        "frame_id", "sequence", "timestamp", "frame",
        "hand_x", "hand_y", "hand_z", "hand_qx", "hand_qy", "hand_qz", "hand_qw",
        "object_x", "object_y", "object_z", "grasp_state", "total_distance_mm"
    ]
    try:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
        print(f"Merged {len(merged_rows)} entries into {output_csv}")
    except Exception as e:
        print(f"Error writing output CSV: {e}")

if __name__ == "__main__":
    main()