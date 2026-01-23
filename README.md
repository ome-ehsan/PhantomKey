# PhantomKey: A Vision-Based Authentication Interface

**A touchless, dynamically scrambling virtual keypad immune to observational attacks.**

## 📖 Executive Summary

Standard PIN authentication suffers from a critical physical vulnerability: **Shoulder Surfing**. An observer (human or camera) can deduce a user's PIN by tracking finger movements or spatial patterns.

**PhantomKey** eliminates this threat by decoupling the spatial location of a button from its numerical value. Using a computer-vision pipeline, it creates a touchless interface where the keypad layout is **randomly scrambled after every input**. Even if an attacker records your hand moving to the top-left corner, that information is useless because the numbers have already shifted.


## 🛠 Core Features

This project demonstrates proficiency in Real-time Computer Vision, State-Machine Logic, and UI/UX Engineering.

1. **Dynamic Scrambling Engine:** The keypad reshuffles the `[0-9]` array instantly upon a valid click, making spatial observational attacks mathematically impossible.


2. **Touchless Interaction:** Uses **MediaPipe Hands** to track 21 skeletal landmarks with sub-millisecond latency, allowing for a strictly non-contact interface.


3. **Secure "Hover" Masking:** The grid is hidden (masked with `*`) by default. Numbers only reveal themselves when the user's cursor hovers over a specific zone, preventing long-range cameras from photographing the layout.


4. **Liveness Detection (Anti-Spoofing):** Integrated **Face Mesh** tracking ensures a real human is present. If the user looks away or leaves, the system enters a `LOCKED` state immediately.


5. **Hybrid Input System:** Supports both Heuristic (Euclidean) and Neural Network (CNN) gesture recognition pipelines.

---

## 🔬 The Engineering Journey: Heuristics vs. Deep Learning

A core objective of this project was to determine the most robust method for detecting a "Click" gesture (Pinch Interaction). I implemented and A/B tested two distinct approaches.

### 1. The Approaches

* **Method A: Euclidean Heuristic (The Math Approach)**
* Calculates the raw distance between the Index Finger Tip (Landmark 8) and Thumb Tip (Landmark 4).
* **Logic:** `if distance < THRESHOLD: click = True`


* **Method B: Convolutional Neural Network (The AI Approach)**
* A custom CNN trained on 650+ collected images of "Pinch" vs. "Open" gestures.
* **Architecture:** Rescaling  Conv2D  MaxPooling  Dense Layers.
* **Pipeline:** Trained locally using TensorFlow; deployed as a quantized `.tflite` model.



### 2. The Benchmark Results

I developed a benchmarking script (`benchmark.py`) to test both models against a validation dataset.

| Metric | Heuristic (Math) | Neural Network (CNN) |
|--------|------------------|----------------------|
| **Accuracy** | **88.54%** | **99.43%** |
| **False Positives** | **78** | 4 |
| **False Negatives** | 2 | 0 |

### 3. The "Accuracy Paradox"

On paper, the Neural Network was flawless. However, in **live production tests**, the Neural Network failed to deliver a usable experience.

* **The Problem:** The NN suffered from **Out-Of-Distribution (OOD)** errors. While it perfectly recognized "Pinches" vs. "Open Hands" from the training set, it struggled with ambiguous intermediate states—specifically the **"Pointing Gesture"** used to hover over buttons.
* **The Failure Mode:** The NN frequently misclassified a "Pointing Finger" as a "Pinch," triggering **False Positives** (phantom clicks). In a security keypad, a False Positive (entering the wrong number) is a catastrophic UX failure, whereas a False Negative (ignoring a click) is merely annoying.

### 4. The Verdict

**I prioritized Reliability over "Smartness".**
Despite accuracy of the Neural Network, I defaulted to the **Heuristic Math Model**. The math approach, while requiring threshold tuning, offered **near-zero False Positives**.


## 🏗 Technical Architecture

The application runs in a high-performance `while True` loop structured into four distinct layers:

1. **Perception Layer (The Eyes):**
* Captures video via OpenCV.
* Runs parallel MediaPipe inference for Hands (Input) and Face Mesh (Security).

2. **State Machine (The Brain):**
* Manages strict application states: `IDLE`  `TRACKING`  `HOVER`  `DEBOUNCE`  `LOCKED` .
* Prevents race conditions and UI bugs.

3. **Logic Layer (The Security):**
* Handles the `scramble()` algorithm and coordinate hit-testing.
* Contains the Fallback Logic (Primary: Math, Secondary: Neural Net).

4. **Rendering Layer (The UI):**
* Draws the dynamic grid and handles the "Hover Masking" visual logic using OpenCV drawing primitives.



## 🚀 Setup & Usage

### Prerequisites

* Python 3.10 or 3.11 (Required for MediaPipe compatibility)
* Webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/phantomkey.git
cd phantomkey

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

```

### Running the App

```bash
python main.py

```

* **Interact:** Raise your hand. The system will track your index finger.
* **Hover:** Move over a `*` to reveal the number.
* **Click:** Pinch your index and thumb to select.
* **Security:** Hide your face or look away to instantly lock the system.

### Running the Benchmark

To replicate the research findings:

```bash
python benchmark.py

```


## 📜 License

This project is open-source and available under the MIT License.

*Developed as a demonstration of robust Computer Vision engineering and Secure UI design.*