import cv2
import os
import mediapipe as mp

# SETUP
INPUT_DIR = "dataset"
OUTPUT_DIR = "dataset_cropped"
CLASSES = ["click", "none"]

# Init MediaPipe (Static Mode)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3 # Lower confidence to catch everything
)

def process():
    print("=== CROPPING DATASET ===")
    
    for label in CLASSES:
        # Create output folders
        in_path = os.path.join(INPUT_DIR, label)
        out_path = os.path.join(OUTPUT_DIR, label)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        files = os.listdir(in_path)
        print(f"Processing '{label}': {len(files)} images...")
        
        count = 0
        for filename in files:
            img_path = os.path.join(in_path, filename)
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            # 1. Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # 2. Calculate Crop Box
                h, w, c = frame.shape
                lm_list = results.multi_hand_landmarks[0].landmark
                
                x_vals = [lm.x for lm in lm_list]
                y_vals = [lm.y for lm in lm_list]
                
                x_min, x_max = int(min(x_vals) * w), int(max(x_vals) * w)
                y_min, y_max = int(min(y_vals) * h), int(max(y_vals) * h)
                
                # Add Padding (Important context!)
                padding = 40
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # 3. Crop & Save
                crop = frame[y_min:y_max, x_min:x_max]
                
                # Check if crop is valid
                if crop.size > 0:
                    cv2.imwrite(os.path.join(out_path, filename), crop)
                    count += 1
        
        print(f"Saved {count} cropped images for '{label}'.")

if __name__ == "__main__":
    process()