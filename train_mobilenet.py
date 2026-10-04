import tensorflow as tf

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained layers
base_model.trainable = False

# Add our own classifier
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(15, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_path = "data/PlantVillage/PlantVillage"

image_paths = []
labels = []

# Collect image paths and class labels
for class_name in os.listdir(data_path):
    class_path = os.path.join(data_path, class_name)

    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Skip the known bad file
            if os.path.basename(image_path) == "svn-r6Yb5c":
                continue

            image_paths.append(image_path)
            labels.append(class_name)

print("Total images:", len(image_paths))
print("Total classes:", len(set(labels)))

# Split data
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

# Convert class names to numbers
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Test images:", len(X_test))
print("Classes:", label_encoder.classes_)

# Load and preprocess images for MobileNetV2
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # MobileNetV2 uses 224x224 images
    image = tf.image.resize(image, (224, 224))

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label


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

# Apply preprocessing
train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Shuffle, batch and prefetch
train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

print("Datasets ready!")

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

# Save the trained model
model.save("plant_disease_mobilenet.keras")

print("MobileNetV2 model trained and saved successfully!")