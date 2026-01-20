import cv2
import mediapipe as mp
import numpy as np

print(f"OpenCV Version: {cv2.__version__}")
print(f"MediaPipe Version: {mp.__version__}")

# Test Webcam 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
else:
    print("SUCCESS: Webcam detected! Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.imshow("Environment Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()