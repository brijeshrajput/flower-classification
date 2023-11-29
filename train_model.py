import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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

def train_and_save_model(train_dir, test_dir, model_save_path, input_shape=(64, 64, 3), batch_size=32, epochs=10, dropout_rate=0.5):
    # Data preparation
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_set = train_datagen.flow_from_directory(train_dir, target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')
    test_set = test_datagen.flow_from_directory(test_dir, target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')

    # Model creation
    model = create_simple_cnn_model(input_shape, len(train_set.class_indices), dropout_rate)

    # Model compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Model training with history to plot the training curves
    history = model.fit(train_set, epochs=epochs, validation_data=test_set)

    # Plot training curves
    plot_training_curves(history)

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
train_and_save_model('data/train', 'data/test', 'flower_classification_model.h5')
