import cv2
import os
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
data_dir = "gesture_data"  # This will be your dataset folder
os.makedirs(data_dir, exist_ok=True)

# Ask for the gesture label (e.g., "inverted_open_palm", "one", "two", "fist", "none")
label = input("Enter gesture label: ").strip().lower()
label_dir = os.path.join(data_dir, label)
os.makedirs(label_dir, exist_ok=True)
print(f"Saving to {label_dir}")

# Webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        # Draw landmarks on frame for visual feedback
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(frame, f"{label} (Samples: {index}) - Press SPACE to save, Q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Collect Gesture Data", frame)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord(' '):  # Spacebar saves the current frame
            cv2.imwrite(os.path.join(label_dir, f"{index}.jpg"), frame)
            print(f"Saved sample {index}")
            index += 1
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()