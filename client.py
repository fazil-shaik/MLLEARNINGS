# client.py
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import base64
from PIL import Image

# Server endpoint
BASE_URL = "http://localhost:8000"

def send_prediction_request(features):
    """Send features and get prediction"""
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"features": features}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']}")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def get_prediction_plot(features):
    """Get prediction plot as image"""
    response = requests.post(
        f"{BASE_URL}/predict-and-plot",
        json={"features": features}
    )
    
    if response.status_code == 200:
        # Display the image
        img = mpimg.imread(BytesIO(response.content), format='png')
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return response.content
    else:
        print(f"Error: {response.status_code}")
        return None

def get_prediction_with_base64_plot(features):
    """Get prediction and base64 encoded plot"""
    response = requests.post(
        f"{BASE_URL}/predict-with-plot-base64",
        json={"features": features}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Prediction: {data['prediction']}")
        
        # Decode and display plot
        plot_bytes = base64.b64decode(data['plot'])
        img = Image.open(BytesIO(plot_bytes))
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# Example usage
if __name__ == "__main__":
    # Example features (adjust based on your model)
    sample_features = [0.5, 0.3, 0.8, 0.2]
    
    # Method 1: Simple prediction
    result = send_prediction_request(sample_features)
    
    # Method 2: Get plot directly
    plot_image = get_prediction_plot(sample_features)
    
    # Method 3: Get prediction with base64 plot
    result_with_plot = get_prediction_with_base64_plot(sample_features)