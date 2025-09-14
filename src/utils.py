import numpy as np
from PIL import Image

def preprocess_image(file_path):
    """Preprocess image for model prediction (grayscale, resize, reshape)"""
    image = Image.open(file_path).convert('L').resize((90, 90))
    image = np.array(image)
    return image.reshape(1, 90, 90, 1)

def predict_traffic_sign(model, classes, file_path):
    """Predict traffic sign from image file path"""
    processed_image = preprocess_image(file_path)
    prediction = model.predict(processed_image)
    prediction_index = np.argmax(prediction, axis=1)[0]
    return classes['Name'][prediction_index]
