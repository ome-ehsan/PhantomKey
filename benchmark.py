import cv2
import os
import time
import mediapipe as mp
import config
from core.logic import SecurityLogic
from core.classifier import GestureClassifier

# setup dirs
DATASET_DIR = "dataset"
RESULTS_FILE = "benchmark_report.txt"

# Initialize Engines
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, # needed for handling independent images
    max_num_hands=1,
    min_detection_confidence=0.5
)
# Init the math algo
math_brain = SecurityLogic()

try:
    nn_brain = GestureClassifier() # Loads models/gesture_model.tflite
    nn_ready = True
except:
    print("WARNING: Neural Net model not found. Benchmark will fail for NN.")
    nn_ready = False

stats = {
    "math": {"correct": 0, "total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0},
    "nn":   {"correct": 0, "total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
}

def run_test():
    print("=== STARTING BENCHMARK: MATH vs NEURAL NET ===")
    
    for label in ["click", "none"]:
        folder_path = os.path.join(DATASET_DIR, label)
        if not os.path.exists(folder_path):
            print(f"Skipping {label} (folder not found)")
            continue
            
        is_positive_class = (label == "click")
        files = os.listdir(folder_path)
        
        print(f"Processing '{label}' class ({len(files)} images)...")
        
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            image = cv2.imread(filepath)
            if image is None: continue
            # getting the landmarks
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            # If no hands, abort both algos
            if not results.multi_hand_landmarks:
                continue 

            hand_landmarks = results.multi_hand_landmarks[0]

            # TEST MATH
            math_result = math_brain.detect_pinch(hand_landmarks) # Returns True/False
            update_stats("math", math_result, is_positive_class)

            # TEST NEURAL NETWORK
            if nn_ready:
                nn_pred = nn_brain.predict(image, hand_landmarks) # Returns "click" or "none"
                nn_result = (nn_pred == "click")
                update_stats("nn", nn_result, is_positive_class)

    print_report()

def update_stats(method, prediction, truth):
    stats[method]["total"] += 1
    
    if prediction == truth:
        stats[method]["correct"] += 1
        if truth: stats[method]["tp"] += 1
        else:     stats[method]["tn"] += 1
    else:
        if prediction and not truth: stats[method]["fp"] += 1 # False Positive
        if not prediction and truth: stats[method]["fn"] += 1 # False Negative

def print_report():
    report = []
    report.append("==========================================")
    report.append("      PHANTOMKEY ACCURACY BENCHMARK       ")
    report.append("==========================================\n")
    
    for method in ["math", "nn"]:
        s = stats[method]
        if s["total"] == 0: continue
        
        accuracy = (s["correct"] / s["total"]) * 100
        name = "Math (Euclidean)" if method == "math" else "Neural Network (CNN)"
        
        report.append(f"Model: {name}")
        report.append(f"------------------------------------------")
        report.append(f"Accuracy:      {accuracy:.2f}%")
        report.append(f"True Positives: {s['tp']} (Clicked correctly)")
        report.append(f"False Positives:{s['fp']} (Clicked when didn't mean to)")
        report.append(f"True Negatives: {s['tn']} (Ignored correctly)")
        report.append(f"False Negatives:{s['fn']} (Ignored a real click)\n")

    report_text = "\n".join(report)
    print(report_text)
    
    with open(RESULTS_FILE, "w") as f:
        f.write(report_text)
    print(f"Report saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_test()