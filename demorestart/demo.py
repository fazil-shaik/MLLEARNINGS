#simple linear regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1000)


df = {
    "sqft":np.random.randint(1000, 8500, 1000),
    "Rooms":np.random.randint(1, 8, 1000),
    "Distance":np.random.randint(10, 90, 1000),
    "Age":np.random.randint(5, 80, 1000),
}

dataset = pd.DataFrame(df)

print(dataset.head(5))


#data selection what to get 

X = dataset.drop(columns=["Age"])
y = dataset["Age"]

# print(X.size,y.size)


#splitting the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Getting the model and trainig on training dataset

LinearModel = LinearRegression()
LinearModel.fit(X_train,y_train)


#model prediction and Values check

y_linear_predict = LinearModel.predict(X_test)

#model eval

print("mean squared error is : ",mean_squared_error(y_test,y_linear_predict))
print("R2 score is : ",r2_score(y_test,y_linear_predict))


#model plotting
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_linear_predict)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values") 
plt.show()