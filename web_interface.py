# web_interface.py
import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
import base64

st.title("ML Model Prediction Interface")

# Input features
st.sidebar.header("Input Features")
features = []
for i in range(4):
    value = st.sidebar.slider(f"Feature {i+1}", 0.0, 1.0, 0.5)
    features.append(value)

if st.button("Predict"):
    # Make prediction request
    response = requests.post(
        "http://localhost:8000/predict-with-plot-base64",
        json={"features": features}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Display prediction
        st.metric("Prediction", f"{data['prediction']:.3f}")
        
        # Display plot
        plot_bytes = base64.b64decode(data['plot'])
        img = Image.open(BytesIO(plot_bytes))
        st.image(img, caption="Prediction Visualization")
    else:
        st.error("Prediction failed!")

# Run with: streamlit run web_interface.py