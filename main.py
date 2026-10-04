import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_path = "data/PlantVillage/PlantVillage"

image_paths = []
labels = []

# Collect image paths and their class labels
for class_name in os.listdir(data_path):
    class_path = os.path.join(data_path, class_name)

    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            image_paths.append(image_path)
            labels.append(class_name)

print("Total images:", len(image_paths))
print("Total classes:", len(set(labels)))

# Check for corrupted/empty images
valid_paths = []
valid_labels = []

for path, label in zip(image_paths, labels):
    try:
        with Image.open(path) as img:
            img.verify()

        if os.path.getsize(path) > 0:
            valid_paths.append(path)
            valid_labels.append(label)

    except:
        print("Skipping bad image:", path)

image_paths = valid_paths
labels = valid_labels

print("Valid images:", len(image_paths))

# 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

# Split remaining 30% into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Test images:", len(X_test))

# Convert class names into numbers
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

print("Classes:", label_encoder.classes_)
print("Encoded labels:", y_train[:10])

# Use a small balanced subset for quick training
import random

random.seed(42)

selected_indices = []

for class_id in sorted(set(y_train)):
    class_indices = [
        i for i, label in enumerate(y_train)
        if label == class_id
    ]

    selected_indices.extend(
        random.sample(class_indices, 50)
    )

X_train = [X_train[i] for i in selected_indices]
y_train = [y_train[i] for i in selected_indices]

print("Quick training images:", len(X_train))

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
)

print(train_dataset)

# Load and preprocess images
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    return image, label

train_dataset = train_dataset.map(load_image)
val_dataset = val_dataset.map(load_image)
test_dataset = test_dataset.map(load_image)

# Batch the datasets
train_dataset = train_dataset.shuffle(750).batch(32)
val_dataset = val_dataset.batch(32)
test_dataset = test_dataset.batch(32)

# Check one batch
for images, labels in train_dataset.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("First label:", labels[0].numpy())

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(256, 256, 3)
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        15,
        activation="softmax"
    )
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=2
)

# Save the trained model
model.save("plant_disease_model.keras")

print("Model saved successfully!")
