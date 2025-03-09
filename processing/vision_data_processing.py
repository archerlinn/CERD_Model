import os
import pyzed.sl as sl
import numpy as np
import imageio

# Input SVO file path and output dataset folder
svo_file_path = "/home/archer/code/c_pour_water_2.svo2"
output_folder = "/home/archer/cerd_data/pour_water_01"
sequence_id = "pour_water_01"  # Modify as needed for your session naming

def create_output_folders(base_folder):
    """
    Creates main output folders for left RGB images, depth maps, and video.
    """
    rgb_left_folder = os.path.join(base_folder, "rgb_left")
    depth_folder = os.path.join(base_folder, "depth")
    video_folder = os.path.join(base_folder, "video")

    os.makedirs(rgb_left_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    return rgb_left_folder, depth_folder, video_folder

def main():
    # Create main output folders and define video output path
    rgb_left_folder, depth_folder, video_folder = create_output_folders(output_folder)
    output_video_path = os.path.join(video_folder, f"{sequence_id}.mp4")

    # Initialize the ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 1280x720 resolution
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA         # Ultra depth mode

    # Open the SVO file
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening SVO file: {err}")
        exit(1)

    # Setup video writer using imageio (requires imageio-ffmpeg)
    fps = 30  # Adjust FPS as needed
    video_writer = imageio.get_writer(output_video_path, fps=fps, codec='libx264')

    # Create sl.Mat objects to hold images and depth
    image_left = sl.Mat()
    depth = sl.Mat()

    # Dictionary to keep track of frame counts for each timestamp (integer part)
    frame_count_by_timestamp = {}

    frame_id = 0
    print("Starting to grab frames...")
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        frame_id += 1

        # Retrieve the left image for video output
        if zed.retrieve_image(image_left, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
            print(f"Warning: Failed to retrieve left image at frame {frame_id}")
            continue

        # Convert the left image data to a proper NumPy array
        left_img = np.array(image_left.get_data())
        if left_img is None or left_img.size == 0:
            print(f"Warning: Invalid left image data at frame {frame_id}, skipping...")
            continue

        # Convert BGRA to RGB: drop alpha channel and swap blue and red channels.
        left_img_rgb = left_img[:, :, :3][:, :, ::-1]

        # Append frame to video writer (imageio expects RGB images)
        video_writer.append_data(left_img_rgb)

        # Get the timestamp (in seconds) for naming
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_seconds()
        timestamp_str = f"{timestamp:.3f}"  # Full timestamp string, e.g. "22.123"
        # Use the integer part of the timestamp to create the subfolder name.
        timestamp_int = int(timestamp)
        subfolder_name = f"timestamp{timestamp_int}"
        
        # Create subfolders under each main category if they don't exist
        left_subfolder = os.path.join(rgb_left_folder, subfolder_name)
        depth_subfolder = os.path.join(depth_folder, subfolder_name)
        os.makedirs(left_subfolder, exist_ok=True)
        os.makedirs(depth_subfolder, exist_ok=True)

        # Update the frame counter for this timestamp
        if timestamp_int not in frame_count_by_timestamp:
            frame_count_by_timestamp[timestamp_int] = 1
        else:
            frame_count_by_timestamp[timestamp_int] += 1
        # Format the frame id for this timestamp as two digits (starting from 01)
        timestamp_frame_id = frame_count_by_timestamp[timestamp_int]
        frame_id_str = f"{timestamp_frame_id:02d}"

        if zed.retrieve_measure(depth, sl.MEASURE.DEPTH) != sl.ERROR_CODE.SUCCESS:
            print(f"Warning: Failed to retrieve depth measure at frame {frame_id}")
            continue

        depth_img = depth.get_data()
        if depth_img is None or depth_img.size == 0:
            print(f"Warning: Invalid depth data at frame {frame_id}, skipping...")
            continue

        # Construct file names with frame id for this timestamp
        left_filename = os.path.join(left_subfolder, f"{sequence_id}_{timestamp_str}_{frame_id_str}.png")
        depth_filename = os.path.join(depth_subfolder, f"{sequence_id}_{timestamp_str}_{frame_id_str}.npy")

        # Save images and depth map
        imageio.imwrite(left_filename, left_img_rgb)
        np.save(depth_filename, depth_img)

        print(f"Saved frame {frame_id} (frame {frame_id_str} for timestamp {timestamp_int}) at timestamp {timestamp_str}")

    # Release resources
    video_writer.close()
    zed.close()
    print("Finished processing SVO file.")

if __name__ == "__main__":
    main()
