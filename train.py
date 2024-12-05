import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Generate or Load Dataset
def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for demonstration. Replace this with your actual dataset.
    """
    np.random.seed(42)
    temperature = np.random.uniform(50, 200, n_samples)  # Simulated temperature sensor
    pressure = np.random.uniform(1, 10, n_samples)      # Simulated pressure sensor
    time_to_failure = 5000 - (temperature * 15 + pressure * 300) + np.random.normal(0, 100, n_samples)
    time_to_failure = np.clip(time_to_failure, 0, None)  # Ensure no negative time-to-failure
    data = pd.DataFrame({
        'temperature': temperature,
        'pressure': pressure,
        'time_to_failure': time_to_failure
    })
    return data

# Generate synthetic dataset
data = generate_synthetic_data()

# Step 2: Split Data into Training and Testing Sets
X = data[['temperature', 'pressure']]
y = data['time_to_failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Step 5: Save the Trained Model
model_filename = "time_to_failure_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_filename}")
