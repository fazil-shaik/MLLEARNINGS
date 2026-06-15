import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Coffee Roast Predictor",
    page_icon="☕",
    layout="wide"
)

# API endpoint (change if deployed)
API_URL = "http://localhost:8000"

# Title and description
st.title("☕ Smart Coffee Roast Predictor")
st.markdown("""
    Predict coffee acidity, sweetness, and body using multiple regression models.
    Adjust the roast parameters below to see how they affect the final cup quality!
""")

# Sidebar for inputs
st.sidebar.header("Roast Parameters")

# Create input sliders
roast_time = st.sidebar.slider(
    "Roast Time (minutes)",
    min_value=8.0,
    max_value=18.0,
    value=11.5,
    step=0.1,
    help="Total duration of the roasting process"
)

temp_ramp = st.sidebar.slider(
    "Temperature Ramp Rate (°C/min)",
    min_value=3.0,
    max_value=8.0,
    value=5.2,
    step=0.1,
    help="How fast the temperature increases per minute"
)

moisture = st.sidebar.slider(
    "Bean Moisture (%)",
    min_value=8.0,
    max_value=14.0,
    value=10.5,
    step=0.1,
    help="Initial moisture content of green beans"
)

density = st.sidebar.slider(
    "Bean Density (g/ml)",
    min_value=0.60,
    max_value=0.80,
    value=0.72,
    step=0.01,
    help="Density of green coffee beans"
)

airflow = st.sidebar.select_slider(
    "Airflow Setting",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    value=6,
    help="Airflow intensity during roasting (1=lowest, 10=highest)"
)

# Model selection
st.sidebar.header("Model Selection")
model_options = ["linear", "lasso", "ridge", "polynomial", "all"]
selected_model = st.sidebar.selectbox(
    "Choose prediction model",
    options=model_options,
    format_func=lambda x: x.capitalize() if x != "all" else "Compare All Models"
)

# Prepare input data
input_data = {
    "roast_time_min": roast_time,
    "temp_ramp_c_min": temp_ramp,
    "moisture_pct": moisture,
    "density_g_ml": density,
    "airflow": airflow
}

# Function to make API call
def get_prediction(model_name):
    try:
        if model_name == "all":
            response = requests.post(f"{API_URL}/predict_all", json=input_data)
        else:
            response = requests.post(f"{API_URL}/predict/{model_name}", json=input_data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}. Make sure FastAPI is running on {API_URL}")
        return None

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Prediction Results")
    
    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Getting predictions from models..."):
            result = get_prediction(selected_model)
            
            if result:
                if selected_model == "all":
                    # Display all model predictions
                    st.subheader("Model Comparison")

                    predictions = result["predictions"]
                    first_value = next(iter(predictions.values()))

                    if isinstance(first_value, dict):
                        predictions_df = pd.DataFrame.from_dict(predictions, orient="index").reset_index()
                        predictions_df = predictions_df.rename(columns={"index": "Model"})
                        predictions_df["Model"] = predictions_df["Model"].str.capitalize()

                        # Create grouped bar chart for multi-metric predictions
                        melted_df = predictions_df.melt(
                            id_vars="Model",
                            var_name="Metric",
                            value_name="Prediction"
                        )
                        fig = px.bar(
                            melted_df,
                            x="Model",
                            y="Prediction",
                            color="Metric",
                            barmode="group",
                            title="Model Predictions Across Major Metrics",
                            labels={"Prediction": "Score", "Metric": "Metric"},
                            range_y=[0, 10]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(predictions_df, use_container_width=True)
                    else:
                        predictions_df = pd.DataFrame({
                            "Model": [name.capitalize() for name in predictions.keys()],
                            "Acidity Prediction": [round(val, 2) for val in predictions.values()]
                        })
                        fig = px.bar(
                            predictions_df,
                            x="Model",
                            y="Acidity Prediction",
                            color="Model",
                            title="Acidity Predictions by Model",
                            range_y=[1, 10]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(predictions_df, use_container_width=True)

                    # Show input parameters
                    with st.expander(" Input Parameters"):
                        st.json(result["input_parameters"])

                else:
                    # Display single model prediction
                    st.metric(
                        label=f"{selected_model.capitalize()} Model - Acidity Prediction",
                        value=f"{result['acidity_prediction']:.2f} / 10",
                        delta=None
                    )
                    
                    # Feature importance
                    if result.get("feature_importance"):
                        st.subheader(" Feature Importance")
                        importance_df = pd.DataFrame(
                            list(result["feature_importance"].items()),
                            columns=["Feature", "Coefficient"]
                        )
                        st.dataframe(importance_df, use_container_width=True)
                
                st.success(f"Prediction completed at {result.get('timestamp', 'N/A')}")
    else:
        st.info("Click the 'Predict' button to see results")

with col2:
    st.header("Model Information")
    
    # Model descriptions
    model_info = {
        "linear": "**Linear Regression**: Baseline model assuming linear relationships",
        "lasso": "**Lasso Regression**: Feature selection through L1 regularization",
        "ridge": "**Ridge Regression**: Handles multicollinearity with L2 regularization",
        "polynomial": "**Polynomial Regression**: Captures non-linear patterns (degree 2)"
    }
    
    for model, description in model_info.items():
        st.markdown(description)
        st.markdown("---")
    
    # Current parameters summary
    st.subheader("Current Parameters")
    st.json(input_data)

# Additional visualizations
st.header(" Roast Profile Visualization")

# Create a simulated temperature profile
if st.checkbox("Show Temperature Profile"):
    # finer resolution for smoother curve
    time_points = list(np.linspace(0, float(roast_time), num=int(roast_time * 4) + 1))

    # Simulate a more realistic roast curve: fast initial ramp that tapers off
    temp_points = []
    for t in time_points:
        base = 25.0
        linear = temp_ramp * t
        # non-linear taper: increases more at start and flattens toward the end
        taper = (1 - (t / roast_time)) * (temp_ramp * 0.8) * np.sqrt(t)
        temp = base + linear + taper
        temp_points.append(float(temp))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=temp_points,
        mode='lines+markers',
        name='Temperature',
        line=dict(color='red', width=2)
    ))
    
    # Add first crack marker
    first_crack = 8.5  # Typical first crack time
    if first_crack < roast_time:
        fig.add_vline(x=first_crack, line_dash="dash", line_color="green",
                      annotation_text="First Crack")
    
    fig.update_layout(
        title="Temperature Profile During Roast",
        xaxis_title="Time (minutes)",
        yaxis_title="Temperature (°C)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Coffee Roast Predictor | Powered by FastAPI & Streamlit | Model Version 1.0.0
    </div>
    """,
    unsafe_allow_html=True
)
