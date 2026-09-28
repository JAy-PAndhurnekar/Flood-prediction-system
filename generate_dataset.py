"""Generates a simple demo flood dataset (not real data)."""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 400

rainfall = np.random.gamma(shape=2.0, scale=20, size=N)
previous_rainfall = np.clip(rainfall * np.random.uniform(0.4, 1.3, N), 0, None)
temperature = np.random.normal(27, 4, N)
humidity = np.clip(40 + rainfall * 0.4 + np.random.normal(0, 8, N), 10, 100)
river_level = np.clip(1 + rainfall * 0.045 + np.random.normal(0, 0.6, N), 0.2, 9)
soil_moisture = np.clip(20 + rainfall * 0.55 + np.random.normal(0, 8, N), 5, 100)

risk_score = (0.03 * rainfall + 0.02 * previous_rainfall + 0.5 * river_level +
              0.03 * soil_moisture + 0.01 * humidity + np.random.normal(0, 1.2, N))
threshold = np.percentile(risk_score, 62)
flood = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    "rainfall": rainfall.round(1),
    "temperature": temperature.round(1),
    "humidity": humidity.round(1),
    "river_level": river_level.round(2),
    "soil_moisture": soil_moisture.round(1),
    "previous_rainfall": previous_rainfall.round(1),
    "flood": flood
})

os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/flood_data.csv", index=False)
print(f"Demo dataset created with {len(df)} rows -> dataset/flood_data.csv")
