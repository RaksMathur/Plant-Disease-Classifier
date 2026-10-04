import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_mobilenet.keras")


model = load_model()


class_names = [
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy"
]

disease_info = {
    "Pepper Bell - Bacterial Spot": {
        "description": "A bacterial disease that causes dark spots on pepper leaves and fruits.",
        "tip": "Remove affected plant parts and avoid overhead watering."
    },
    "Pepper Bell - Healthy": {
        "description": "The pepper plant appears healthy based on the model prediction.",
        "tip": "Continue regular watering, sunlight, and plant care."
    },
    "Potato - Early Blight": {
        "description": "A fungal disease that commonly causes dark spots and yellowing on potato leaves.",
        "tip": "Remove severely affected leaves and avoid keeping foliage wet for long periods."
    },
    "Potato - Late Blight": {
        "description": "A disease that can cause dark lesions on potato leaves and stems.",
        "tip": "Remove affected plant material and maintain good air circulation."
    },
    "Potato - Healthy": {
        "description": "The potato plant appears healthy based on the model prediction.",
        "tip": "Continue regular plant care and monitor the leaves for changes."
    },
    "Tomato - Bacterial Spot": {
        "description": "A bacterial disease that can cause small dark spots on tomato leaves and fruits.",
        "tip": "Avoid overhead watering and remove severely affected plant parts."
    },
    "Tomato - Early Blight": {
        "description": "A fungal disease that commonly produces dark spots with a target-like pattern on tomato leaves.",
        "tip": "Remove affected leaves and improve airflow around the plant."
    },
    "Tomato - Late Blight": {
        "description": "A disease that can cause dark lesions on tomato leaves and stems.",
        "tip": "Remove affected plant material and avoid prolonged leaf wetness."
    },
    "Tomato - Leaf Mold": {
        "description": "A fungal disease that commonly affects tomato leaves, especially under humid conditions.",
        "tip": "Improve air circulation and reduce excess humidity around the foliage."
    },
    "Tomato - Septoria Leaf Spot": {
        "description": "A fungal disease that causes numerous small spots on tomato leaves.",
        "tip": "Remove affected leaves and avoid splashing water onto foliage."
    },
    "Tomato - Spider Mites": {
        "description": "Tiny pests that can cause speckling, discoloration, and damage to tomato leaves.",
        "tip": "Inspect the underside of leaves and maintain appropriate plant care."
    },
    "Tomato - Target Spot": {
        "description": "A fungal disease that produces circular spots with darker centers on tomato leaves.",
        "tip": "Remove affected leaves and improve airflow around the plant."
    },
    "Tomato - Yellow Leaf Curl Virus": {
        "description": "A viral disease that can cause yellowing, curling, and reduced growth in tomato plants.",
        "tip": "Monitor the plant for symptoms and control insect vectors such as whiteflies."
    },
    "Tomato - Mosaic Virus": {
        "description": "A viral disease that can cause mottled or mosaic-like patterns on tomato leaves.",
        "tip": "Remove severely affected plants and maintain good hygiene when handling plants."
    },
    "Tomato - Healthy": {
        "description": "The tomato plant appears healthy based on the model prediction.",
        "tip": "Continue regular watering, sunlight, and routine plant monitoring."
    }
}


# Page title
st.title("🌿 Plant Disease Classifier")
st.write("Upload a plant leaf image to predict its disease.")


# Upload image
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Leaf Image", width=400)

    # Resize image
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    # Display result
    st.subheader("Prediction")

    st.success(
        f"🌱 Disease: {class_names[predicted_index]}"
    )

    st.info(
        f"🎯 Confidence: {confidence:.2f}%"
    )

    # Display disease information
    st.subheader("About the Prediction")

    st.write(disease_info[class_names[predicted_index]]["description"])

    st.subheader("Care Tip")

    st.write(disease_info[class_names[predicted_index]]["tip"])