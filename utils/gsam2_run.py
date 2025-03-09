import os
import cv2
import json
import torch
import numpy as np
import open3d as o3d
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import matplotlib.pyplot as plt  # For colormap
import imageio

# --- Configuration Parameters ---
TEXT_PROMPT = "bottle. cup. tilted bottle."  # Adjust as needed.
# Updated folder structure using the new dataset folder (pour_water_03)
RGB_FOLDER = "/home/archer/cerd_data/pour_water_01/rgb_left"
DEPTH_FOLDER = "/home/archer/cerd_data/pour_water_01/depth"
OUTPUT_MASK_FOLDER = Path("/home/archer/cerd_data/pour_water_01/masks")
OUTPUT_MASK_FOLDER.mkdir(parents=True, exist_ok=True)
POINT_CLOUD_OUTPUT_FOLDER = Path("/home/archer/cerd_data/pour_water_01/point_clouds")
POINT_CLOUD_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.45
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DUMP_JSON_RESULTS = True

# --- Load Camera Intrinsics ---
calib_file = "/home/archer/cerd_data/calibration.json"
with open(calib_file, "r") as f:
    calib = json.load(f)
intrinsics = calib["intrinsic_left"]
fx = intrinsics["fx"]
fy = intrinsics["fy"]
cx = intrinsics["cx"]
cy = intrinsics["cy"]
print(f"Loaded intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# --- Initialize Models ---

sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# --- Helper Functions ---

def extract_frame_id(filename):
    base = os.path.basename(filename)
    # Remove the file extension and split by underscore.
    parts = os.path.splitext(base)[0].split("_")
    # Use the last part as the frame id.
    return parts[-1]


def extract_sequence_id(filename):
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) >= 3:
        # Join the first three parts to form the sequence id.
        return "_".join(parts[:3])
    return "default_sequence"


def extract_timestamp(filename):
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) >= 4:
        timestamp = parts[3]
        print(f"Extracted timestamp: {timestamp} from filename: {base}")
        return timestamp
    print(f"Failed to extract timestamp from {base}")
    return None

def sanitize_label(label):
    """Sanitize object label for filenames (e.g., replace spaces with underscores)."""
    return label.lower().replace(" ", "_")

def process_grounded_sam2_for_image(img_path):
    """Process one RGB image with Grounded SAM2 and save segmentation masks with object labels."""
    timestamp = extract_timestamp(img_path)
    if timestamp is None:
        print(f"Could not extract timestamp from {img_path}")
        return None

    sequence_id = extract_sequence_id(img_path)

    # Assumes the filename already includes a frame id; we use it as is.
    frame_id = extract_frame_id(img_path)
    print(f"Processing image for timestamp: {timestamp}, frame id: {frame_id}")
    # Create subfolder for this timestamp under masks.
    ts_int = int(float(timestamp))
    mask_subfolder = OUTPUT_MASK_FOLDER / f"timestamp{ts_int}"
    mask_subfolder.mkdir(exist_ok=True)
    
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)
    
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    h, w, _ = image_source.shape
    print(f"Original boxes (normalized): {boxes}")
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    print(f"Converted boxes: {input_boxes}")
    
    # Get segmentation masks using SAM2.
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        if masks.shape[1] == 3:
            masks = masks[:, 0, :, :]
        elif masks.shape[1] == 1:
            masks = masks.squeeze(1)
    
    if len(labels) != len(masks):
        print(f"WARNING: Number of detected labels ({len(labels)}) does not match number of masks ({len(masks)})")
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        label_cleaned = sanitize_label(label)
        mask_uint8 = (mask.astype(np.uint8)) * 255
        _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        out_filename = os.path.join(mask_subfolder, f"{sequence_id}_{timestamp}_{frame_id}_mask_{label_cleaned}.png")
        cv2.imwrite(out_filename, mask_uint8)
        print(f"Saved segmentation mask: {out_filename}")
    
    return timestamp

def generate_point_cloud_from_mask(depth_map, mask, fx, fy, cx, cy):
    mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))
    print("DEBUG: Mask shape after resize:", mask.shape)
    ys, xs = np.where(mask > 0)
    print("DEBUG: Number of non-zero pixels in mask:", len(ys))
    if len(ys) == 0:
        print("WARNING: Mask has no non-zero pixels!")
        return np.empty((0, 3))
    depths = depth_map[ys, xs]
    valid = depths > 0
    num_valid = np.count_nonzero(valid)
    print("DEBUG: Number of valid depth pixels (depth > 0):", num_valid)
    if num_valid == 0:
        print("WARNING: No valid depth values found for masked region!")
        return np.empty((0, 3))
    xs = xs[valid]
    ys = ys[valid]
    depths = depths[valid]
    X = (xs - cx) * depths / fx
    Y = (ys - cy) * depths / fy
    Z = depths
    points = np.vstack((X, Y, Z)).T
    print("DEBUG: Number of points generated:", points.shape[0])
    return points

def add_depth_colors_to_pcd(pcd):
    points = np.asarray(pcd.points)
    if points.size == 0:
        return pcd
    depths = points[:, 2]
    depth_min = depths.min()
    depth_max = depths.max()
    norm_depths = (depths - depth_min) / (depth_max - depth_min + 1e-8)
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm_depths)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_point_cloud(points, filename, voxel_size=0.01):
    if points.shape[0] == 0:
        print(f"WARNING: No points to save for {filename}. Skipping point cloud generation.")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print("DEBUG: Point cloud before downsampling has", np.asarray(pcd.points).shape[0], "points")
    
    try:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    except RuntimeError as e:
        print(f"WARNING: Skipping point cloud {filename} due to voxel_down_sample error: {e}")
        return
    
    print("DEBUG: Point cloud after voxel downsampling has", np.asarray(pcd.points).shape[0], "points")
    
    # Remove statistical outliers.
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = add_depth_colors_to_pcd(pcd)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename} with {np.asarray(pcd.points).shape[0]} points")


def process_point_cloud_for_timestamp(timestamp):
    ts_int = int(float(timestamp))
    depth_subfolder = os.path.join(DEPTH_FOLDER, f"timestamp{ts_int}")
    mask_subfolder = os.path.join(OUTPUT_MASK_FOLDER, f"timestamp{ts_int}")
    # Create a corresponding point cloud subfolder.
    point_cloud_subfolder = os.path.join(POINT_CLOUD_OUTPUT_FOLDER, f"timestamp{ts_int}")
    os.makedirs(point_cloud_subfolder, exist_ok=True)
    
    # Select .npy files that include the given timestamp in their name.
    depth_files = [f for f in os.listdir(depth_subfolder) if f.endswith(".npy") and f"_{timestamp}_" in f]
    if not depth_files:
        print(f"No depth files found for timestamp {timestamp} in {depth_subfolder}")
        return

    for depth_file in depth_files:
        # Extract the sequence_id and frame_id from the depth file's name.
        # Expected depth file format: "pour_water_01_1740349025.000_01.npy"
        sequence_id = extract_sequence_id(depth_file)
        frame_id = extract_frame_id(depth_file)
        
        depth_filepath = os.path.join(depth_subfolder, depth_file)
        print(f"DEBUG: Loading depth map from {depth_filepath}")
        try:
            depth_map = np.load(depth_filepath)
            print("DEBUG: Loaded depth map with shape:", depth_map.shape)
        except Exception as e:
            print(f"ERROR: Could not load depth map {depth_filepath}: {e}")
            continue

        mask_prefix = f"{sequence_id}_{timestamp}_{frame_id}_"
        segmentation_files = [f for f in os.listdir(mask_subfolder) if f.startswith(mask_prefix) and f.endswith(".png")]
        if not segmentation_files:
            print(f"No segmentation masks found for depth file {depth_file}")
            continue

        for seg_file in segmentation_files:
            mask_path = os.path.join(mask_subfolder, seg_file)
            print(f"DEBUG: Processing segmentation mask {mask_path}")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not load mask {mask_path}")
                continue
            mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))
            print("DEBUG: Mask shape after resize:", mask.shape)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            points = generate_point_cloud_from_mask(depth_map, mask, fx, fy, cx, cy)
            if points.shape[0] == 0:
                print(f"WARNING: No points generated for mask {mask_path}")
                continue

            # Save point cloud: use a larger voxel size to avoid the voxel_size error.
            ply_filename = os.path.join(point_cloud_subfolder, seg_file.replace('.png', '.ply'))
            save_point_cloud(points, ply_filename, voxel_size=0.05)



# --- Batch Processing Pipeline ---

# Step 1: Process all RGB left images with Grounded SAM2 to generate segmentation masks.

rgb_subfolders = [os.path.join(RGB_FOLDER, d) for d in os.listdir(RGB_FOLDER) if os.path.isdir(os.path.join(RGB_FOLDER, d))]
all_rgb_files = []
for folder in rgb_subfolders:
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    all_rgb_files.extend(files)
all_rgb_files.sort()
timestamps = []
for rgb_file in all_rgb_files:
    print(f"Processing image: {rgb_file}")
    ts = process_grounded_sam2_for_image(rgb_file)
    if ts is not None:
        timestamps.append(ts)
timestamps = sorted(set(timestamps))
print("Timestamps processed:", timestamps)

# Step 2: For each timestamp, generate 3D point clouds using corresponding depth maps and segmentation masks.
for ts in timestamps:
    print(f"Processing point cloud for timestamp: {ts}")
    process_point_cloud_for_timestamp(ts)
