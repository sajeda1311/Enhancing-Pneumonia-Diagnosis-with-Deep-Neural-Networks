import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("models/chest_xray_cnn.h5")

# Input image path from command line
img_path = sys.argv[1]

# Load and preprocess image
img_height, img_width = 180, 180

img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)[0][0]  # Get scalar
confidence = prediction if prediction > 0.5 else 1 - prediction
label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

# Output
print(f"\nPrediction: {label}")
print(f"Confidence: {confidence * 100:.2f}%")