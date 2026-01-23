import tensorflow as tf
import cv2
import numpy as np

class GestureClassifier:
    def __init__(self, model_path="models/gesture_model.tflite"):
        # loading and allocating tensors 
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # loading labels (0=click, 1=none based on training output)
        self.labels = ['click', 'none'] 

    def predict(self, frame, hand_landmarks):
        h, w, c = frame.shape
        
        # getting the bounding box around hands 
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h) # this is done as medipipe coords are normalized 
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y

        # adding to not crop fingers off 
        padding = 40
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # extracting crop
        hand_crop = frame[y_min:y_max, x_min:x_max]
        
        # return none if crop is empty
        if hand_crop.size == 0:
            return "none"

        # 3resizing and normalizing 
        img = cv2.resize(hand_crop, (128, 128))
        img = img.astype(np.float32) / 255.0 # Normalize to [0,1]
        img = np.expand_dims(img, axis=0)    # Add batch dimension [1, 128, 128, 3]

        #runnig inference 
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        # taking output of the highest score 
        prediction_index = np.argmax(output_data)
        result = self.labels[prediction_index]
        
        return result