import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time

model = tf.keras.models.load_model("landmark_model.h5")
labels = np.load("labels.npy")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.40
last_spoken = ""
last_time = 0
SPEAK_DELAY = 2

cv2.namedWindow("Sign Language Translator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sign Language Translator", 1200, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            data = np.array(landmarks).flatten().reshape(1, -1)

            prediction = model.predict(data, verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > CONFIDENCE_THRESHOLD:
                label = labels[class_id]
            else:
                label = "Unknown"

            cv2.putText(frame,
                        f"{label} ({confidence:.2f})",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.5,
                        (0, 0, 255),
                        5)

            mp_draw.draw_landmarks(frame,
                                   hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            current_time = time.time()

            if label != "Unknown" and label != last_spoken:
                if current_time - last_time > SPEAK_DELAY:
                    threading.Thread(target=speak, args=(label,), daemon=True).start()
                    last_spoken = label
                    last_time = current_time

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()