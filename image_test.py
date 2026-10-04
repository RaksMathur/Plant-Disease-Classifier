from PIL import Image
import os
import numpy as np

folder_path = "data/PlantVillage/PlantVillage/Tomato_healthy"

images = os.listdir(folder_path)

image_path = os.path.join(folder_path, images[0])

image = Image.open(image_path)

print("Image path:", image_path)
print("Image size:", image.size)
print("Image mode:", image.mode)

image_array = np.array(image)

print("Array shape:", image_array.shape)
print("First pixel:", image_array[0, 0])

normalized_image = image_array / 255.0

print("Original pixel:", image_array[0, 0])
print("Normalized pixel:", normalized_image[0, 0])
