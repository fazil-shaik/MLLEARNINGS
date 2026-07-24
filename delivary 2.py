import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
data = {
    "traffic_level": ["High","Low","Medium","Low","High","Medium","Low","High","Low","Medium"],
    "actual_delivery_time": [42,20,55,30,28,45,27,60,18,70]
}
df = pd.DataFrame(data)
traffic_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

df["traffic_numeric"] = df["traffic_level"].map(traffic_mapping)
X = df[["traffic_numeric"]]   # single feature
y = df["actual_delivery_time"]
model = LinearRegression()
model.fit(X, y)
predicted = model.predict(np.array([[3]]))  # Predict for "High traffic"

print("")
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted delivery time for HIGH traffic:", predicted[0])
