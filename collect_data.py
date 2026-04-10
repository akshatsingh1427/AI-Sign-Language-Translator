import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

label = input("Enter label: ")
os.makedirs(f"dataset2/{label}", exist_ok=True)

count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            np.save(f"dataset2/{label}/{count}.npy", np.array(landmarks))
            count += 1

    cv2.putText(frame, f"Samples: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):#imp
        break

cap.release()
cv2.destroyAllWindows()
