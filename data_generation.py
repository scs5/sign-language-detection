import cv2
import os
import shutil

LABELS = ['A', 'B', 'C']
NUM_IMAGES = 100
DATA_DIR = './data'


def clear_data():
    """ Clears any existing data in the data directory. """
    # Remove directory if it exists
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

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


def capture_images():
    """ Captures images for each label. """
    # Open webcam
    cap = cv2.VideoCapture(0)
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
            if key == ord(' '):
                break

        # Save images to data directory
        print(f"Capturing images for label '{label}'...")
        for i in range(NUM_IMAGES):
            ret, frame = cap.read()
            cv2.imshow('Webcam', frame)
            img_filename = os.path.join(DATA_DIR, label, f"{label}_{i}.png")
            cv2.imwrite(img_filename, frame)
            cv2.waitKey(50)

    # Close the camera
    cap.release()
    cv2.destroyAllWindows()
    print("Image capture complete.")


if __name__ == "__main__":
    clear_data()
    capture_images()