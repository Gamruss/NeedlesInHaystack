import cv2 as cv
import numpy as np
import os

try:
    # Change the working directory to the folder containing the script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Path to the folder containing images to search for (needles)
    needles_folder = "needles"
    # Path to the folder containing images to search in (haystacks)
    haystacks_folder = "haystacks"
    # Path to the folder to save the results
    results_folder = "results"

    # Create the results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Get the list of files (images) in the needles folder
    needle_files = os.listdir(needles_folder)

    # Loop through each image in the haystacks folder
    for haystack_file in os.listdir(haystacks_folder):
        # Read the haystack image
        haystack_img = cv.imread(os.path.join(haystacks_folder, haystack_file))
        if haystack_img is None:
            print(f"Error: Unable to read the image {haystack_file}")
            continue

        # Remove alpha channel if present
        if haystack_img.shape[2] == 4:
            haystack_img = cv.cvtColor(haystack_img, cv.COLOR_BGRA2BGR)

        # Loop through each needle image
        for needle_file in needle_files:
            # Read the needle image
            needle_img = cv.imread(os.path.join(needles_folder, needle_file))
            if needle_img is None:
                print(f"Error: Unable to read the image {needle_file}")
                continue

            # Remove alpha channel if present
            if needle_img.shape[2] == 4:
                needle_img = cv.cvtColor(needle_img, cv.COLOR_BGRA2BGR)

            # Perform template matching
            result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

            # Get the best match position and confidence
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

            # If the best match confidence is above threshold, consider it a match
            threshold = 0.8
            if max_val >= threshold:
                print(f"Found needle {needle_file} in haystack {haystack_file}.")

                # Draw rectangle around the matched region with increased height
                needle_w = needle_img.shape[1]
                needle_h = needle_img.shape[0]
                top_left = max_loc
                bottom_right = (top_left[0] + needle_w, top_left[1] + 2 * needle_h)  # Increased height

                # Create a mask to keep only the region of interest
                mask = np.zeros_like(haystack_img)
                mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255

                # Apply the mask to the haystack image
                result_img = cv.bitwise_and(haystack_img, mask)

                # Set rectangle color to pink
                pink_color = (203, 192, 255)  # BGR values for pink
                cv.rectangle(result_img, top_left, bottom_right,
                             color=pink_color, thickness=2, lineType=cv.LINE_AA)

                # Save the result with the name of the needle image
                result_filename = os.path.splitext(needle_file)[0] + "_" + haystack_file
                result_path = os.path.join(results_folder, result_filename)
                cv.imwrite(result_path, result_img)

            else:
                print(f"Needle {needle_file} not found in haystack {haystack_file}.")

    # Wait for the user to press Enter before closing the OpenCV windows
    input("Press Enter to exit...")

except Exception as e:
    print(f"An error occurred: {e}")
    # Wait for the user to press Enter before closing the window in case of error
    input("Press Enter to exit...")
