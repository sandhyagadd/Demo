import cv2
import numpy as np
import json
import math

# Compare frames function
def compare_frames(frame1, frame2, areas_of_interest, spillage_threshold):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Create a copy of the original frame for drawing bounding boxes
    frame_copy = frame1.copy()

    # Keep track of detected chemicals in each region
    chemical_detected_regions = [False] * len(areas_of_interest)

    # Iterate over the areas of interest
    for idx, area_of_interest in enumerate(areas_of_interest):
        x, y, w, h = area_of_interest

        # Crop the frames to the area of interest
        crop1 = gray1[y:y+h, x:x+w]
        crop2 = gray2[y:y+h, x:x+w]

        # Calculate the absolute difference between the cropped frames
        diff = cv2.absdiff(crop1, crop2)

        # Apply threshold to highlight the differences
        threshold = 15
        _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Calculate the percentage of changed pixels
        total_pixels = w * h
        changed_pixels = np.count_nonzero(thresholded)
        percentage_changed = (changed_pixels / total_pixels) * 100

        # Find contours of the differences
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the changed regions in red
        for contour in contours:
            x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
            if math.sqrt(w_cont + h_cont) > 1:
                # Check if the red box is detected in the blue region
                if x_cont >= 0 and y_cont >= 0 and x_cont + w_cont <= w and y_cont + h_cont <= h:
                    # Crop the corresponding regions from the original color frames
                    crop_frame1 = frame1[y:y+h, x:x+w]
                    crop_frame2 = frame2[y:y+h, x:x+w]

                    # Convert cropped regions to HSV to check for brown or black color
                    hsv1 = cv2.cvtColor(crop_frame1, cv2.COLOR_BGR2HSV)
                    hsv2 = cv2.cvtColor(crop_frame2, cv2.COLOR_BGR2HSV)

##                    # Define the lower and upper bounds of brown color in HSV
##                    lower_brown = np.array([5, 50, 50])
##                    upper_brown = np.array([30, 255, 200])
##
##                    # Define the lower and upper bounds of black color in HSV
##                    lower_black = np.array([0, 0, 0])
##                    upper_black = np.array([180, 255, 30])
##
##                    # Create masks to detect brown and black colors in both frames
##                    mask1_brown = cv2.inRange(hsv1, lower_brown, upper_brown)
##                    mask2_brown = cv2.inRange(hsv2, lower_brown, upper_brown)
##                    mask1_black = cv2.inRange(hsv1, lower_black, upper_black)
##                    mask2_black = cv2.inRange(hsv2, lower_black, upper_black)

                    # Define the lower and upper bounds of red color in HSV
                    lower_red = np.array([0, 100, 100])
                    upper_red = np.array([30, 255, 255])

                    # Create masks to detect red color in both frames
                    mask1 = cv2.inRange(hsv1, lower_red, upper_red)
                    mask2 = cv2.inRange(hsv2, lower_red, upper_red)
                    

                    # Check if both brown and black colors are present in both frames
                    if (cv2.countNonZero(mask1) > 0 or cv2.countNonZero(mask2) > 0):
                       # Check if the percentage of changed pixels exceeds the spillage threshold
                        if percentage_changed > spillage_threshold:
                            # Set the flag to True for chemical spillage detection
                            chemical_detected_regions[idx] = True
##                            print("True")

                # Draw bounding box in red
                cv2.rectangle(frame_copy, (x + x_cont, y + y_cont), (x + x_cont + w_cont, y + y_cont + h_cont), (0, 0, 255), 2)

        # Draw area of interest rectangles in blue
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Print the percentage of changed pixels for the current area of interest
        print(f"Area of interest {idx + 1}: {area_of_interest}")
        print(f"Percentage of changed pixels: {percentage_changed}%")

    # Print which area of interest chemical is detected
    for idx, detected in enumerate(chemical_detected_regions):
        area_name = f"region {idx + 1}"
        if detected:
            print(f"Chemical Detected in {area_name}")
        else:
            print(f"No Chemical Detected in {area_name}")

    return frame_copy

# Image based
def image_difference_1(json_file, download_path, spillage_threshold):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    video1 = data['video1']
    video2 = data['video2']
    areas_of_interest = data['areas_of_interest']
    download_path = data['download_path']

    # Open the images
    frame1 = cv2.imread(video1, 1)
    frame2 = cv2.imread(video2, 1)

    # resize frames
    frame1 = cv2.resize(frame1, (640,480))
    frame2 = cv2.resize(frame2, (640,480))

    # Compare frames and get the result with bounding boxes
    result_frame = compare_frames(frame1, frame2, areas_of_interest, spillage_threshold)

    # Display the result frame with bounding boxes
    cv2.imshow("Difference Image", result_frame)

    # Save the difference image
    img_path = f"{download_path}/difference_image.png"
    cv2.imwrite(f"{download_path}/image_1.png", frame1)
    cv2.imwrite(f"{download_path}/image_2.png", frame2)
    cv2.imwrite(img_path, result_frame)

    # Print the path of the saved difference image
    print(f"Difference image saved as: {img_path}")

# Example usage with JSON file
json_file = "video_comparison.json"
download_path = "."
spillage_threshold = 5  # Define your desired spillage threshold percentage(pixel threshold)
image_difference_1(json_file, download_path, spillage_threshold)
