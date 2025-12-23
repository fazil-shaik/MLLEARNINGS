# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import statsmodels.api as sm


# data = {
#     'Size_sqft': [800, 1000, 1200, 1500, 1800, 2000, 2300, 2600, 3000],
#     'Bedrooms': [1, 2, 2, 3, 3, 3, 4, 4, 5],
#     'Age_years': [20, 15, 10, 8, 5, 6, 4, 3, 2],
#     'Distance_km': [12, 10, 8, 7, 6, 5, 4, 3, 2],
#     'Parking': [0, 0, 1, 1, 1, 1, 1, 1, 1],
#     'Price_lakhs': [40, 55, 75, 95, 120, 135, 160, 185, 220]
# }

# df = pd.DataFrame(data)

# print(df.describe())

# sns.pairplot(df)
# plt.show()

# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.show()

# # Log transform target to handle skewness
# df['Log_Price'] = np.log(df['Price_lakhs'])

# X = df[['Size_sqft', 'Bedrooms', 'Age_years', 'Distance_km', 'Parking']]
# y = df['Log_Price']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = LinearRegression()
# model.fit(X_train, y_train)


# y_pred_train = model.predict(X_train)
# y_pred_test = model.predict(X_test)

# print("Train R²:", r2_score(y_train, y_pred_train))
# print("Test R²:", r2_score(y_test, y_pred_test))

# print("MAE:", mean_absolute_error(y_test, y_pred_test))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))


# coefficients = pd.DataFrame({
#     'Feature': X.columns,
#     'Coefficient': model.coef_
# })

# print(coefficients)


# X_vif = sm.add_constant(X)

# vif_data = pd.DataFrame()
# vif_data['Feature'] = X_vif.columns
# vif_data['VIF'] = [
#     variance_inflation_factor(X_vif.values, i)
#     for i in range(X_vif.shape[1])
# ]

# print(vif_data)






