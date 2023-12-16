import os
import pickle
import mediapipe as mp
import cv2
from utils.config import *


def extract_features():
    """ Extracts positions of different parts of the hand to be used as features. """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    data = []
    labels = []
    for label in os.listdir(DATA_DIR):
        print("Extracting features for '{}'".format(label))
        for img_path in os.listdir(os.path.join(DATA_DIR, label)):
            hand_features = []

            # Convert image from BGR to RGB
            img = cv2.imread(os.path.join(DATA_DIR, label, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect hand features from image
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                        hand_features.append(x)
                        hand_features.append(y)
                data.append(hand_features)
                labels.append(label)

    # Save data to picke file
    with open(DATA_PICKLE_FN, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Finished extracting features.")


if __name__ == '__main__':
    extract_features()