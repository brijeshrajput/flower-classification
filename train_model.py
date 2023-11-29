import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data preparation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('data/train', target_size=(64, 64), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('data/test', target_size=(64, 64), batch_size=32, class_mode='categorical')

# Model creation (using a simple CNN as an example)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(units=len(train_set.class_indices), activation='softmax'))

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
model.fit(train_set, epochs=10, validation_data=test_set)

# Save the model for future use
model.save('flower_classification_model.h5')
