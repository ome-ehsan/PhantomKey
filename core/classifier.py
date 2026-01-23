import tensorflow as tf
import cv2
import numpy as np

class GestureClassifier:
    def __init__(self, model_path="models/gesture_model.tflite"):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.labels = ['click', 'none']
            print(f"SUCCESS: Loaded model from {model_path}")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load model. {e}")
            raise e

    def predict(self, frame, hand_landmarks):
        h, w, c = frame.shape
        
        #Bounding Box
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y

        # Add Padding
        padding = 40
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Extract Crop
        hand_crop = frame[y_min:y_max, x_min:x_max]
        if hand_crop.size == 0: return "none"

        # Resize to 128x128
        img = cv2.resize(hand_crop, (128, 128))
        
        #  Convert BGR (OpenCV) to RGB (TensorFlow)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype(np.float32)
        
        #  Add batch dimension -> Shape (1, 128, 128, 3)
        img = np.expand_dims(img, axis=0)

        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        
        # Raw Scores (Logits)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        


        prediction_index = np.argmax(output_data)
        return self.labels[prediction_index]