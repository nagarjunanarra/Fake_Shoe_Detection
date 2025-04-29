from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_model.h5')
model = load_model(MODEL_PATH)

# Retrieve class labels from the model (ensure these match your training labels)
class_indices = {'Fake Air Force': 0, 'Fake Jordan': 1, 'Ori Jordan': 2, 'Other Class': 3}
index_to_class = {v: k for k, v in class_indices.items()}

def predict_image(uploaded_image):
    """
    Preprocesses the uploaded image and returns a prediction from the trained model.
    Converts numpy data types to native Python types for FastAPI compatibility.
    """
    # Preprocess the image (resize to model's expected input size and normalize)
    img = uploaded_image.resize((150, 150))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

    # Perform prediction
    predictions = model.predict(img_array)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])  # Convert to int
    predicted_class = index_to_class[predicted_class_index]
    confidence = float(np.max(predictions)) * 100  # Convert to native float and percentage

    # Return JSON-serializable response
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2)  # Rounded to two decimal places
    }
