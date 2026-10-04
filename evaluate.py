import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load trained MobileNetV2 model
model = tf.keras.models.load_model("plant_disease_mobilenet.keras")

data_path = "data/PlantVillage/PlantVillage"

image_paths = []
labels = []

# Collect image paths and labels
for class_name in os.listdir(data_path):
    class_path = os.path.join(data_path, class_name)

    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            if os.path.basename(image_path) == "svn-r6Yb5c":
                continue

            image_paths.append(image_path)
            labels.append(class_name)

# Same split as training
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

# Convert labels to numbers
label_encoder = LabelEncoder()

label_encoder.fit(y_train)

y_test = label_encoder.transform(y_test)

# Load and preprocess test images
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label

# Create test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Evaluate model
loss, accuracy = model.evaluate(test_dataset)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")