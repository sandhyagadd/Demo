import cv2
import os

def video_to_frames(input_video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(input_video_path)

    # Get the frames per second (fps) of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize variables
    frame_count = 0
    success = True

    while success:
        # Read a frame from the video
        success, frame = video.read()

        # If the frame was read successfully, save it as an image
        if success:
            frame_count += 1
            frame_name = f"{frame_count:04d}.jpg"  # Naming the frames with 4-digit zero-padded numbers
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)

    # Release the video capture object
    video.release()

    print(f"Video converted to frames: {frame_count} frames saved in {output_folder}")

# Example usage
input_video_path = r"C:\Sandhya\Development\chemical_detection\New folder\All_files\videos\Chemical Spill .mp4"  # Replace with the path to your input video
output_folder = r"C:\Sandhya\Development\chemical_detection\New folder\All_files\New folder" # Folder where frames will be saved
video_to_frames(input_video_path, output_folder)
