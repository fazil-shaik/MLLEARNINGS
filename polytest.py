import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

np.random.seed(42)
#created a data using dataframe using numpy random things
df = pd.DataFrame({
    'area_sqft':np.random.randint(400,2500,500),
    'bedrooms':np.random.randint(1,5,500),
    'distance_near_metro_km':np.round(np.random.uniform(0.5,15,500),1),
    'floors':np.random.randint(0,20,500),
    'building_age':np.random.randint(1,20,500)
})

#define features which effects the prediction!(y=mx+b) b=5000 x= rent_per_sqft = 50
# rent_per_sqft_squared = 0.01
# rent_per_bedroom = 3000
# # rent_per_bedroom_squrred = 0.35
# rent_reduction_per_km = 800
# rent_per_floor = 300
# rent_per_age_buidling = 300



base_rent = 5000
rent_per_sqft = 50
rent_per_sqft_squared = 0.01
rent_per_bedroom = 3000
# rent_per_bedroom_squrred = 0.35
rent_reduction_per_km = 800
rent_per_floor = 300
rent_per_age_buidling = 300

#calcilating rent 

df['rent']=(
    base_rent
    +rent_per_sqft*df['area_sqft']
    +rent_per_sqft_squared*(df['area_sqft'] ** 2)
    +rent_per_bedroom*df['bedrooms']
    -rent_reduction_per_km*df['distance_near_metro_km']
    +rent_per_floor*df['floors']
    -rent_per_age_buidling*df['building_age']
)

print(df.head(5))

#prepare the data
X = df['area_sqft'].values.reshape(-1,1)
y = df['rent'].values

#train the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#create a model now
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)

#predict the model

y_prediction = linear_model.predict(X_test)

print("\n Linear Regression")
print(f"Rent = {linear_model.intercept_:.0f} + {linear_model.coef_[0]:.0f} * Area")

#polynomial regression!
poly = PolynomialFeatures(degree=2, include_bias=False)


#data prep
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


#model creation
poly_model = LinearRegression()
poly_model.fit(X_train_poly,y_train)


print("\n polynomial Features")
print(f"Orginal values: Area = {X_train[0][0]}")
print(f"After values:{X_train_poly[0][0]:.0f}, Area Squared = {X_train_poly[0][1]: .0f}")

#predict data from poly!

y_predict_poly = poly_model.predict(X_test_poly)


#output
print("\n Polynomial Regression")
print(f"rent = {poly_model.intercept_:.0f} + {poly_model.coef_[0]:.0f} * Area +{poly_model.coef_[1]:.4f} * Area square ")



#plot the data

X_plot = np.linspace(X.min(),X.max(),100).reshape(-1,1)
x_plot_poly = poly.transform(X_plot)


plt.figure(figsize=(10,6))

plt.scatter(X,y,alpha=0.3,label='Actual data')

#linear regressin scatter
plt.plot(X_plot,linear_model.predict(X_plot),color='red',linewidth=2,label='Linear regression')

#plot polynomial regression
plt.plot(X_plot,poly_model.predict(x_plot_poly),color='green',linewidth=2,label='Polynomial Regression')

#show
plt.xlabel('Area per sqft')
plt.ylabel('Rent RS:')
plt.title('Linear vs Poly')
plt.legend()
plt.show()


#accuracy between linear vs poly

r2_linear = r2_score(y_test,y_prediction)
r2_poly = r2_score(y_test,y_predict_poly)


print(f"\n Accuracy comparison")
print(f"Linear regression Accuracy: {r2_linear:.4f}")
print(f"Polynomial regression Accuracy: {r2_poly:.4f}")