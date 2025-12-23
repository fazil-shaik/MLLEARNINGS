import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

res = np.random.seed(42)


#create data


#pandas-dataframes-make whatevrer data we have put it in table 
# #apartment lit 500
# data = {
#     "Apartment_ID": [101, 102, 103, 104, 105],
#     "Apartment_Name": ["Green Heights", "Sky Residency", "Lake View", "Urban Nest", "Sunrise Homes"],
#     "City": ["Bangalore", "Hyderabad", "Chennai", "Pune", "Bangalore"],
#     "Bedrooms": [2, 3, 2, 1, 3],
#     "Bathrooms": [2, 2, 1, 1, 3],
#     "Area_sqft": [1200, 1500, 1100, 650, 1800],
#     "Rent_per_month": [25000, 32000, 22000, 15000, 40000],
#     "Furnished": ["Semi", "Fully", "Unfurnished", "Fully", "Semi"],
#     "Parking": [True, True, False, False, True]
# }
# df = pd.DataFrame(data)

df = pd.DataFrame({
    'area_sqft':np.random.randint(400,2500,50),
    'bed_rooms':np.random.randint(1,5,50),
    'distance_to_metro_INkm':np.round(np.random.uniform(0.5,10,50),1),
    'floor_num':np.random.randint(1,10,50),
    'building_age':np.random.randint(1,30,50),
})

# print(df.head(5))

#lets see what effects if rent changes

base_rent = 5000
rent_sq_ft = 15
rent_per_bedroom = 3000
rent_per_km = 800
rent_per_floor = 200
rent_reduction_per_building_age = 300

#y = mx+b


#calcualted the rent of the data
df['rent'] = (
    base_rent
    +
    rent_sq_ft*df['area_sqft']
    +
    rent_per_bedroom*df['bed_rooms']
    -
    rent_per_km*df['distance_to_metro_INkm']
    +
    rent_per_floor*df['floor_num']
    -
    rent_reduction_per_building_age*df['building_age']
)

print(df.head())


X = df[['area_sqft']]
y = df[['rent']]

#splitting data 80% train and 20% test
X_tain,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#model creation
model = LinearRegression()

#train model
model.fit(X_tain,y_train)


#predict data while testing only!

y_pred = model.predict(X_test)

#pprint ouputs using df


print(f"\n Simple Linear regression")
# print(f"Rent={model.intercept_:02f} + {model.coef_:02f} * Area")
print(f"Rent = {model.intercept_} + {model.coef_} * Area")


#plot the data
plt.scatter(X_test,y_test,alpha=0.5,label='Actual data')
plt.plot(X_test.sort_values('area_sqft'),model.predict(X_test.sort_values('area_sqft')),color="red",label="predicted value")
plt.xlabel('Area in sqft')
plt.ylabel('Rent prediction in $')
plt.title('Simple Linear Regression')
plt.show()