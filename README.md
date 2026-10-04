

# 🌿 Plant Disease Classifier

A deep learning-based web application that predicts plant diseases from leaf images using **MobileNetV2 Transfer Learning** and **Streamlit**.

The model can classify leaf images into **15 different plant health/disease categories** covering pepper, potato, and tomato plants.

## 🚀 Features

* Upload a plant leaf image
* Predict the disease/health condition
* Display prediction confidence
* Show a short description of the predicted condition
* Provide a basic care tip
* Simple and interactive Streamlit interface

## 🧠 Model

The project uses **MobileNetV2** with ImageNet pretrained weights.

Instead of training a CNN completely from scratch, transfer learning was used to take advantage of features already learned by MobileNetV2.

### Model Architecture

* MobileNetV2 (pretrained on ImageNet)
* Global Average Pooling
* Dense layer with 128 neurons
* Output layer with 15 classes
* Softmax activation for classification

The pretrained MobileNetV2 base was frozen during training.

## 📊 Dataset

The model was trained using the **PlantVillage dataset**.

After removing one invalid image, the dataset contained **20,638 valid images** across 15 classes.

### Dataset Split

| Dataset    | Images |
| ---------- | -----: |
| Training   | 14,446 |
| Validation |  3,096 |
| Testing    |  3,096 |

### Classes

1. Pepper Bell - Bacterial Spot
2. Pepper Bell - Healthy
3. Potato - Early Blight
4. Potato - Late Blight
5. Potato - Healthy
6. Tomato - Bacterial Spot
7. Tomato - Early Blight
8. Tomato - Late Blight
9. Tomato - Leaf Mold
10. Tomato - Septoria Leaf Spot
11. Tomato - Spider Mites
12. Tomato - Target Spot
13. Tomato - Yellow Leaf Curl Virus
14. Tomato - Mosaic Virus
15. Tomato - Healthy

## 📈 Model Performance

The trained model achieved approximately:

**Test Accuracy: 91.70%**

The model was evaluated on a separate test set containing 3,096 images.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2
* NumPy
* Pandas
* Pillow
* Scikit-learn
* Streamlit

## 📁 Project Structure

```text
Plant Disease Classifier/
│
├── data/
│   └── PlantVillage/
│
├── venv/
│
├── main.py
├── train_mobilenet.py
├── evaluate.py
├── predict.py
├── app.py
├── plant_disease_mobilenet.keras
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Plant-Disease-Classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install tensorflow numpy pandas pillow scikit-learn streamlit
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔍 How It Works

1. The user uploads a plant leaf image.
2. The image is converted to RGB format.
3. The image is resized to **224 × 224 pixels**.
4. MobileNetV2 preprocessing is applied.
5. The trained model predicts probabilities for all 15 classes.
6. The class with the highest probability is selected.
7. The application displays the predicted condition and confidence score.
8. A short description and care tip are also shown.

## 💡 Why MobileNetV2?

A CNN trained from scratch required a long training time on CPU.

MobileNetV2 was chosen because it is a lightweight pretrained architecture that can provide good performance through transfer learning while requiring much less training than building a complete CNN from scratch.


This project is intended for **educational and demonstration purposes**. The prediction should not be considered a professional agricultural diagnosis.
