from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
# House data (area, bedrooms, bathrooms)
X = np.array([
    [800, 2, 1],
    [1000, 3, 2],
    [1200, 3, 2],
    [1500, 4, 3],
    [1800, 4, 3]
])

y = np.array([50, 65, 70, 90, 105])  # price in lakhs


model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

model.fit(X, y)
print("Ridge model trained successfully")


y_prediction = model.predict(X=X)
#predicting the price
new_house = np.array([[2000,5,4]])
predicted_Pirce = model.predict(new_house)

print(f"Predicted price for the new house: {predicted_Pirce[0]:.2f} lakhs") 

effeciency = r2_score(y_true=y,y_pred=y_prediction)

print(f"efficiency of model : {effeciency}")

print(f"mean squared error is {mean_squared_error(y,y_prediction)}")

print(f"Root mean squared error is {root_mean_squared_error(y,y_prediction)}")
