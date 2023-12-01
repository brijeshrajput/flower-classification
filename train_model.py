import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def create_simple_cnn_model(input_shape, num_classes, dropout_rate=0.5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(units=num_classes, activation='softmax')
    ])
    return model

def visualize_sample_images(train_set, train_datagen, num_samples=9):
    # Visualize resized images from the first batch
    sample_images, sample_labels = next(train_set)

    # Plot a few sample images
    plt.figure(figsize=(15, 5 * (num_samples // 3 + 1)))
    for i in range(min(num_samples, sample_images.shape[0])):  # Plot up to 9 images
        plt.subplot(num_samples // 3 + 1, 6, 2 * i + 1)
        plt.imshow(sample_images[i])
        plt.title(f"Original\nClass: {sample_labels[i].argmax()}", color="red")  # Display the class (assuming one-hot encoding)
        plt.axis("off")

        train_datagen.rescale = None
        augmented_image = next(train_datagen.flow(np.expand_dims(sample_images[i], 0), batch_size=1))[0]
        plt.subplot(num_samples // 3 + 1, 6, 2 * i + 2)
        plt.imshow(augmented_image)
        plt.title("Augmented", color="green")
        plt.axis("off")

    plt.show()

def train_and_save_model(data_dir, model_save_path, input_shape=(64, 64, 3), batch_size=32, epochs=10, dropout_rate=0.5, test_size=0.2):
    # Data preparation
    datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # Split data into training and test sets
    all_files = []
    all_labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            files = [os.path.join(label_path, file) for file in os.listdir(label_path)]
            all_files.extend(files)
            all_labels.extend([label] * len(files))

    df = pd.DataFrame({'file_path': all_files, 'label': all_labels})
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])

    train_set = datagen.flow_from_dataframe(train_df, x_col='file_path', y_col='label', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')
    test_set = datagen.flow_from_dataframe(test_df, x_col='file_path', y_col='label', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')

    # Visualize resized images from the first batch
    visualize_sample_images(train_set, datagen)
    datagen.rescale = 1./255

    # Model creation
    model = create_simple_cnn_model(input_shape, len(train_set.class_indices), dropout_rate)

    # Model compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Display model summary
    model.summary()

    # Model training with history to plot the training curves
    history = model.fit(train_set, epochs=epochs, validation_data=test_set)

    # Plot training curves
    plot_training_curves(history)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_set)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # Save the model
    model.save(model_save_path)

def plot_training_curves(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Executing
train_and_save_model('data','flower_classification_model.h5')
