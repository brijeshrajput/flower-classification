# Load the trained model
model = load_model('flower_classification_model.h5')

# Get the validation data
X_val, y_val = get_validation_data()

# Evaluate the model on the validation data
results = model.evaluate(X_val, y_val)

# Print the accuracy
accuracy = results['accuracy']
print('Accuracy:', accuracy)