import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('flower_classification_model.h5')

# Function to preprocess live frames
def preprocess_live_frame(frame):
    # Resize the frame to match the input size of the model
    frame = cv2.resize(frame, (64, 64))
    # Convert the frame to a format suitable for prediction
    frame = image.img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    # Normalize pixel values
    frame = frame / 255.0
    return frame

# Function to predict live frames
def predict_live_frame(processed_frame):
    # Make predictions using the loaded model
    predictions = model.predict(processed_frame)
    class_index = np.argmax(predictions)
    # class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    class_labels = ['Bougainvillea', 'Bright Eyes', 'Cape Jasmine', 'Chandni', 'Dhalia', 'Hibiscus', 'Marigold', 'Pink Oleander', 'Rose', 'Tecoma']
    predicted_class = class_labels[class_index]
    return predicted_class

# Function to capture live frames from the camera
def capture_live_frames():
    cap = cv2.VideoCapture(0)  # 0 indicates the default camera

    while True:
        ret, frame = cap.read()

        # Process the frame (resize, preprocess, etc.)
        processed_frame = preprocess_live_frame(frame)

        # Make predictions
        predicted_class = predict_live_frame(processed_frame)

        # Display the frame with the predicted class
        cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Flower Detection', frame)

        # Wait for 1 millisecond and check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start live flower detection
capture_live_frames()
