import mediapipe as mp
import cv2
import numpy as np
import os
from generate_csv import get_image_list

def process_images():
    # mediapipe code
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Run the functions to the get the image directory tree and connection dictionary
    image_dict = get_image_list()

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Iterating over all the folders
        for folder, image_names in image_dict.items():
            print(f"Processing folder: {folder}")
            # Iterating over all the images in folder
            for image_name in image_names:
                # Read image with OpenCV & flip
                image = cv2.imread(f"./images/{folder}/{image_name}")
                image = cv2.flip(image, 1)

                # Process the image
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.multi_hand_landmarks:
                    continue
                
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )
                # Create Image
                cv2.imwrite(f"./processed_images/{folder}/{image_name}", cv2.flip(image, 1))

if __name__ == "__main__":
    process_images()
