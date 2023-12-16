import pickle
import cv2
import mediapipe as mp
import numpy as np
from utils.config import *


def realtime_inference():
    """ Predicts signs in real-time using camera. """
    model_dict = pickle.load(open(MODEL_FN, 'rb'))
    model = model_dict['model']

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    num_iteration = 0
    while True:
        hand_features = []
        x_positions = []
        y_positions = []
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Exit on escape or window close
        key = cv2.waitKey(1)
        if (key == 27 or (cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1 and num_iteration > 1)):
            break

        # Skip iteration if 0 or 2 hands are seen
        results = hands.process(frame_rgb)
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 1:
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            num_iteration += 1
            continue

        # Draw hand marks on camera
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Add hand marks to features
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_positions.append(x)
                y_positions.append(y)

                hand_features.append(x)
                hand_features.append(y)

        # Form bounding box of hand marks
        H, W, _ = frame.shape
        x1 = int(min(x_positions) * W) - 10
        y1 = int(min(y_positions) * H) - 10
        x2 = int(max(x_positions) * W) + 10
        y2 = int(max(y_positions) * H) + 10

        # Predict sign
        prediction = model.predict([np.asarray(hand_features)])
        predicted_character = prediction[0]

        # Display on camera
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('Webcam', frame)
        num_iteration += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    realtime_inference()