import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import time
import pytz
ist = pytz.timezone('Asia/Kolkata')

# Load the trained model
model_filename = "time_to_failure_model.pkl"
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Initialize the Streamlit app
st.set_page_config(page_title="Real-Time Machine Dashboard", layout="wide")
st.title("🌟 Real-Time Machine Monitoring Dashboard")
st.subheader("📊 Predicting Time to Failure using Temperature and Pressure Data")

# Initialize session state for persistent data
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["timestamp", "temperature", "pressure", "time_to_failure"])
    st.session_state.last_update = datetime.now()

# Function to generate synthetic data
def generate_synthetic_data():
    """
    Generates synthetic sensor data (temperature, pressure) and predicts time to failure.
    """
    current_time = datetime.now(ist)
    temperature = np.random.uniform(50, 200)  # Simulated temperature sensor
    pressure = np.random.uniform(1, 10)      # Simulated pressure sensor
    features = pd.DataFrame([[temperature, pressure]], columns=["temperature", "pressure"])
    time_to_failure = model.predict(features)[0]
    return {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": round(temperature, 2),
        "pressure": round(pressure, 2),
        "time_to_failure": round(time_to_failure, 2)
    }

# Initialize with one row of data if DataFrame is empty
if st.session_state.data.empty:
    initial_data = generate_synthetic_data()
    st.session_state.data = pd.DataFrame([initial_data])
    st.session_state.last_update = datetime.now()

# Check if enough time has passed (10 seconds) before updating data
current_time = datetime.now()
time_since_last_update = (current_time - st.session_state.last_update).total_seconds()

# Update data and trigger rerun after 10 seconds
if time_since_last_update > 10:
    new_data = generate_synthetic_data()
    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_data])], ignore_index=True)
    st.session_state.last_update = current_time
    st.experimental_rerun()  # Triggers page refresh after 10 seconds

# Display live sensor values and prediction in colorful tiles
st.subheader("📟 Latest Sensor Readings and Prediction")
if not st.session_state.data.empty:
    latest_data = st.session_state.data.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white;">Temperature</h2>
                <h1 style="color: white;">{latest_data['temperature']}°C</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="background-color: #2196F3; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white;">Pressure</h2>
                <h1 style="color: white;">{latest_data['pressure']} bar</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div style="background-color: #FF5733; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white;">Time to Failure</h2>
                <h1 style="color: white;">{latest_data['time_to_failure']} sec</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div style="background-color: #6A1B9A; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white;">Timestamp</h2>
                <h1 style="color: white;">{latest_data['timestamp']}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.write("Waiting for data...")

# Add artificial delay
time.sleep(10)
