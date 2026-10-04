import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("plant_disease_mobilenet.keras")
# Class names
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Change this to the path of your test image
image_path = r"C:\Users\user\Desktop\Plant Disease Classifier\data\PlantVillage\Tomato_healthy\0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555.JPG"
# Load image
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

image_array = np.array(image, dtype=np.float32)

# MobileNetV2 preprocessing
image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

print("Input shape:", image_array.shape)

# Prediction
predictions = model.predict(image_array)

predicted_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_index] * 100

print("Predicted disease:", class_names[predicted_index])
print("Confidence:", round(confidence, 2), "%")