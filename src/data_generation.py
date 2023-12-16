import cv2
import os
from utils.config import *


def clear_data():
    """ Clears any existing data in the data directory. """
    # Remove directory if it exists
    if os.path.exists(DATA_DIR):
        os.system('rmdir /S /Q "{}"'.format(DATA_DIR))

    # Recreate empty directories for each label
    os.makedirs(DATA_DIR)
    for label in LABELS:
        label_dir = os.path.join(DATA_DIR, label)
        os.makedirs(label_dir)
    print("Data directory cleared.")


def display_message(image, message):
    """ Displays a message on the camera feed. """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(message, font, font_scale, font_thickness)[0]
    text_position = ((image.shape[1] - text_size[0]) // 2, 50)
    cv2.putText(image, message, text_position, font, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA)


def close_webcam(cap, message=''):
    cap.release()
    cv2.destroyAllWindows()
    print(message)


def capture_images():
    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    for label in LABELS:
        # Wait for the user to press the spacebar to start capturing images
        message = f"Label '{label}'. Spacebar to start."
        while True:
            ret, frame = cap.read()
            display_message(frame, message)
            cv2.imshow('Webcam', frame)
            key = cv2.waitKey(1) & 0xFF

            # Space to start recording
            if key == ord(' '):
                break
            # Exit on escape or window close
            elif key == 27 or cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
                close_webcam(cap, "Image capture aborted.")
                return

        # Save images to data directory
        print(f"Capturing images for label '{label}'...")
        for i in range(NUM_IMAGES):
            ret, frame = cap.read()
            cv2.imshow('Webcam', frame)
            img_filename = os.path.join(DATA_DIR, label, f"{label}_{i}.png")
            cv2.imwrite(img_filename, frame)
            cv2.waitKey(50)

            key = cv2.waitKey(1) & 0xFF
            # Exit on escape or window close
            if key == 27 or cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
                close_webcam(cap, "Image capture aborted.")
                return

    # Close the camera
    close_webcam(cap, "Image capture complete.")


if __name__ == "__main__":
    clear_data()
    capture_images()