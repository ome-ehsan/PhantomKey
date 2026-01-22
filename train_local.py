import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import os

# 1. SETUP
# Define paths and parameters
DATA_DIR = pathlib.Path("dataset")
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16 # Small batch size for small dataset

print(f"TensorFlow Version: {tf.__version__}")
print("Loading data...")

# 2. LOAD DATA
# We use a 80/20 split: 80% for training, 20% for testing
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}") 
# EXPECTED OUTPUT: ['click', 'none'] (Order might vary)

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 3. BUILD MODEL (CNN)
num_classes = len(class_names)

model = models.Sequential([
    # Input & Rescaling
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Data Augmentation (Makes model robust to angles)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # Convolutions (The "Vision" part)
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Classification (The "Decision" part)
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes) # Output layer
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. TRAIN
print("Starting Training...")
epochs = 15 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 5. EXPORT TO TFLITE
print("Converting to TFLite...")
# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
if not os.path.exists("models"):
    os.makedirs("models")
    
with open('models/gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save the labels too, so we know which ID is "click"
with open('models/labels.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')

print("SUCCESS: Model saved to models/gesture_model.tflite")
print(f"Class Mapping: {class_names}")