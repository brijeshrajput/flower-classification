import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model('flower_classification_model.h5')

# Function to preprocess image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Function to make predictions for multiple images
def predict_flowers(image_paths):
    predictions_list = []
    for image_path in image_paths:
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        class_index = np.argmax(predictions)
        # You might need to define your own mapping of class indices to class labels
        class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Update with your actual class labels
        class_label = class_labels[class_index]
        predictions_list.append(class_label)
    return predictions_list

# Function to display images in a grid with predicted names
def display_images(image_paths, predicted_classes):
    num_images = len(image_paths)
    # Dynamically determine grid size based on the number of images
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(num_images):
        ax = axes[i // cols, i % cols]
        img = image.load_img(image_paths[i], target_size=(64, 64))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(predicted_classes[i])
    plt.show()

# Example usage
image_folder_path = 'testing/'  # Update with the folder containing your images
image_paths = [os.path.join(image_folder_path, img) for img in os.listdir(image_folder_path) if img.endswith('.jpg')]

predicted_classes = predict_flowers(image_paths)
display_images(image_paths, predicted_classes)

