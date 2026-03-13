import cv2
import mediapipe as mp
import numpy as np

# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
# We only track 1 hand to keep it simple, with a 70% confidence threshold
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 2. Initialize the webcam
cap = cv2.VideoCapture(0)

# Variables to hold the previous finger position so we can draw continuous lines
px, py = 0, 0

# 3. Create a blank canvas to draw on
# We assume a standard webcam resolution of 640x480. 
canvas = np.zeros((480, 640, 3), np.uint8)

print("Starting Air Canvas... Press 'c' to clear the screen, and 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a selfie-view display (like a mirror)
    frame = cv2.flip(frame, 1)

    # MediaPipe requires RGB images, but OpenCV uses BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
    result = hands.process(rgb_frame)

    # 4. If a hand is detected, find the index finger
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            # Landmark 8 is the tip of the index finger
            index_finger_tip = hand_landmarks.landmark[8]

            # Convert normalized coordinates (0.0 to 1.0) to actual pixel coordinates
            h, w, c = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # If this is the first frame we see the finger, set previous to current
            if px == 0 and py == 0:
                px, py = cx, cy

            # Draw a line on our blank canvas from the previous point to the current point
            cv2.line(canvas, (px, py), (cx, cy), (255, 0, 255), 5) # Purple line, thickness 5

            # Draw a circle on the live camera feed so you know where your "pen" is
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # Update previous coordinates for the next frame
            px, py = cx, cy
    else:
        # If no hand is detected, reset previous coordinates so the line breaks
        px, py = 0, 0

    # 5. Merge the drawing canvas with the live camera feed
    # Convert canvas to grayscale, create a mask, and invert it
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    # Black out the region on the camera frame where the drawing exists
    frame = cv2.bitwise_and(frame, img_inv)
    # Layer the colored canvas on top of the blacked-out region
    frame = cv2.bitwise_or(frame, canvas)

    # Show the final result
    cv2.imshow("Air Canvas", frame)

    # 6. Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Press 'q' to quit
        break
    elif key == ord('c'): # Press 'c' to clear the drawing
        canvas = np.zeros((480, 640, 3), np.uint8)

# Cleanup
cap.release()
cv2.destroyAllWindows()