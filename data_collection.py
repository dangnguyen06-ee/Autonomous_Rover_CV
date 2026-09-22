import cv2
import os
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
data_dir = "gesture_data"
os.makedirs(data_dir, exist_ok=True)

# Ask for the gesture label
label = input("Enter gesture label: ").strip().lower()
label_dir = os.path.join(data_dir, label)
os.makedirs(label_dir, exist_ok=True)

# Find next available index to avoid overwriting existing files
existing = [int(f.split('.')[0]) for f in os.listdir(label_dir) 
            if f.endswith('.jpg') and f.split('.')[0].isdigit()]
index = max(existing) + 1 if existing else 0
print(f"Saving to {label_dir}")
print(f"Found {len(existing)} existing samples — starting from index {index}")

# Webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"{label} (Saved: {index}) - SPACE save, D undo, Q quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Collect Gesture Data", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord(' '):
            cv2.imwrite(os.path.join(label_dir, f"{index}.jpg"), frame)
            print(f"Saved sample {index}")
            index += 1
        elif key == ord('d'):
            last_file = os.path.join(label_dir, f"{index - 1}.jpg")
            if os.path.exists(last_file):
                os.remove(last_file)
                index -= 1
                print(f"Deleted sample {index} — next save reuses this index")
            else:
                print("Nothing to delete")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()