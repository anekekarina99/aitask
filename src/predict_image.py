# src/predict_image.py

import tensorflow as tf # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
import matplotlib.pyplot as plt
import sys

# Load the saved model
def load_model(model_path='simpan_resnet_model.h5'):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    return model

# Preprocess the image for prediction
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)  # Load and resize image
    img_array = image.img_to_array(img)                      # Convert to array
    img_array = np.expand_dims(img_array, axis=0)            # Add batch dimension
    img_array /= 255.0                                       # Normalize
    return img_array

# Predict the class of the image
def predict_image(model, img_path, class_names):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]      # Get the predicted class index
    confidence = predictions[0][predicted_class]             # Confidence score for predicted class
    return class_names[predicted_class], confidence

# Plot the image with prediction
def display_prediction(img_path, predicted_class, confidence):
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
    plt.axis('off')
    plt.show()

# Main function for running the prediction
def main(img_path, model_path='simpan_resnet_model.h5'):
    # Define the class names (modify as per your dataset)
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']

    # Load the model
    model = load_model(model_path)

    # Predict and display the result
    predicted_class, confidence = predict_image(model, img_path, class_names)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")
    display_prediction(img_path, predicted_class, confidence)

if __name__ == "__main__":
    # Ensure the image path is passed as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <path_to_image>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    main(img_path)
