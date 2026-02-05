import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree

df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Admission%20Chance.csv')
print(df.head())


# Lets implement Linear regression

X = df[['CGPA']]
y = df['Research']

#model training and splitting
X_Train,X_test,y_Train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection and fitting
model = LinearRegression()
model.fit(X_Train,y_Train)

#prediction
y_prediction = model.predict(X_test)

print(f"mean square error is {mean_squared_error(y_prediction,y_test)}")

# print(f"All predictions are {y_prediction}")

# plotting

plt.figure(figsize=(10,6))
plt.scatter(X,y,label='Orginal data')
# plt.plot(X_test,y_test,label='test data plotting')
plt.plot(y_prediction,label='Predicted data',color='red',marker='*')
plt.xlabel('CGPA got in annually')
plt.ylabel('Research done or not')
plt.show()



#decision Tree regression check now
y = df['Chance of Admit ']
X = df.drop(['Serial No','Chance of Admit '],axis=1)


#traintest split check one
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=2529)

#model selection and checking decision tree
dtr=DecisionTreeRegressor(max_depth=3, random_state=2529)

#fitting the model
dtr.fit(X_train,y_train)

#model score check 
dtr.score(X_train,y_train)


#predictions 

y_predicdtr = dtr.predict(X_test)


print(f"All the predictions we got {y_predicdtr}")

print(f"mse of dtr {mean_squared_error(y_test,y_predicdtr)}")
print(f"r2_Score of dtr {r2_score(y_test,y_predicdtr)}")

fig,ax = plt.subplots(figsize=(15,10))
final=DecisionTreeRegressor(max_depth=3, random_state=2529)
final.fit(X_train,y_train)
plot_tree(final,feature_names=X.columns,filled=True);
plt.show()