import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Create the Dataset
data = {
    'pages': [5, 12, 3, 8, 15, 4, 10, 6, 20, 7,
              9, 2, 14, 5, 11, 3, 8, 18, 6, 13,
              4, 16, 7, 10, 2, 12, 5, 9, 15, 6],

    'deadline_days': [14, 7, 21, 5, 10, 18, 6, 12, 8, 15,
                      4, 25, 9, 11, 7, 20, 6, 5, 14, 8,
                      16, 6, 10, 4, 22, 7, 13, 5, 9, 11],

    'rate_inr': [8000, 22000, 5000, 18000, 28000, 6500, 19000, 10000, 35000, 11000,
                 20000, 3500, 25000, 9000, 21000, 5500, 17000, 38000, 9500, 24000,
                 7000, 32000, 13000, 23000, 4000, 22500, 8500, 19500, 29000, 11500]
}

df = pd.DataFrame(data)

# Step 3: Explore the Data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

#step-4 viualize data




#step-5 prepare training data
X = df[['pages', 'deadline_days']]
y = df['rate_inr']

#step 6 train test and split the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#step 7 model selection and training
model = LinearRegression()
model.fit(X_train,y_train)

# Step 8: Check Model Parameters
print("\nModel Parameters:")
print(f"Coefficients: Pages = {model.coef_[0]:.2f}, Deadline = {model.coef_[1]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
#step 9: Prediction of output
y_prediction = model.predict(X_test)
print("\nPredictions vs Actual:")
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_prediction.round()})
print(results)

# Step 10: Model Accuracy
from sklearn.metrics import r2_score
score = r2_score(y_test, y_prediction)
print(f"\nModel Accuracy (R² Score): {score:.2f}")

# Step 11: Answer Priya's Question
new_project = [[10, 5]]  # 10 pages, 5 days deadline
predicted_rate = model.predict(new_project)
print(f"\nPriya's Question: 10-page website, 5-day deadline")
print(f"Recommended Rate: ₹{predicted_rate[0]:,.0f}")
