#!/usr/bin/env python3
import csv
import re

# ------------- CONFIGURATION -------------
GRASP_CSV_PATH = "/home/archer/cerd_data/pour_water_01/grasp.csv"  # Path to your grasp CSV file

# Define the start endpoint and end endpoint:
# For example, to update from (timestamp=1740349022.000, frame=22) to (timestamp=1740349032.000, frame=25)
START_TIMESTAMP = 1740349023.000
START_FRAME     = 22
END_TIMESTAMP   = 1740349023.000
END_FRAME       = 29
NEW_STATE       = "0"             # New grasp state ("0" or "1")

# Regex pattern to extract sequence, timestamp, frame, and hand_side from hand_id.
# Expected hand_id format: "pour_water_01_1740349022.000_22_0_hand_3d"
hand_id_pattern = re.compile(
    r'^(?P<sequence>.+?)_(?P<timestamp>\d+\.\d+?)_(?P<frame>\d+?)_(?P<hand>[01])_hand_3d$'
)

def extract_ts_frame(hand_id):
    """Extracts (timestamp, frame) from a hand_id string."""
    m = hand_id_pattern.match(hand_id)
    if m:
        try:
            ts = float(m.group("timestamp"))
            frame = int(m.group("frame"))
            return ts, frame
        except ValueError:
            return None, None
    return None, None

def update_grasp_state():
    """Reads the CSV, sorts rows by (timestamp, frame), and updates rows between the start and end endpoints."""
    rows = []
    try:
        with open(GRASP_CSV_PATH, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                # Extract (timestamp, frame) from the hand_id column
                hand_id = row.get("hand_id", "")
                ts, frame = extract_ts_frame(hand_id)
                # Save the ordering tuple as a helper (it will be removed later)
                row["_order"] = (ts if ts is not None else 0, frame if frame is not None else 0)
                rows.append(row)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Sort rows by the extracted (timestamp, frame)
    rows.sort(key=lambda r: r["_order"])

    # Define the endpoints as tuples
    start_tuple = (START_TIMESTAMP, START_FRAME)
    end_tuple = (END_TIMESTAMP, END_FRAME)

    updated_count = 0
    for row in rows:
        ts, frame = row["_order"]
        if ts is None:
            continue
        # Update if (timestamp, frame) falls between start_tuple and end_tuple (inclusive)
        if start_tuple <= (ts, frame) <= end_tuple:
            row["grasp"] = NEW_STATE
            updated_count += 1

    # Remove the helper "_order" field before writing back
    for row in rows:
        row.pop("_order", None)

    try:
        with open(GRASP_CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Updated grasp state for {updated_count} entries in '{GRASP_CSV_PATH}'.")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

def main():
    update_grasp_state()

if __name__ == "__main__":
    main()
