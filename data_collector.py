import cv2
import os
import time

# Setup Folders
DATA_DIR = "dataset"
CLASSES = ["none", "click"]

for label in CLASSES:
    path = os.path.join(DATA_DIR, label)
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    cap = cv2.VideoCapture(0)
    print("=== DATA COLLECTOR ===")
    print("Press '0' to save a 'NONE' (Open Hand) image.")
    print("Press '1' to save a 'CLICK' (Pinch) image.")
    print("Press 'q' to quit.")
    
    counts = {label: len(os.listdir(os.path.join(DATA_DIR, label))) for label in CLASSES}

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        
        # Display counts
        display = frame.copy()
        cv2.putText(display, f"None: {counts['none']} | Click: {counts['click']}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Data Collector", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Save Logic
        label = None
        if key == ord('0'):
            label = "none"
        elif key == ord('1'):
            label = "click"
            
        if label:
            timestamp = int(time.time() * 1000)
            filename = os.path.join(DATA_DIR, label, f"{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            counts[label] += 1
            print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()