# import pandas as pd

# df = pd.read_csv("./customer.csv")

# print(df.head())

# print(df.tail())

# print(df.shape)

# print(df.info())

# print("unique values count are : ",df.nunique().count())

# print(df.describe())

# print(df.isnull().sum())

# new_dataframe = df.drop(
#     columns=['customer_id','first_name','last_name']
# )

# print(new_dataframe.head())

# print("Detect values: ",new_dataframe.isnull().sum())


# missing_rows = new_dataframe[new_dataframe.isnull().any(axis=1)]

# print('Missing rows are: ',missing_rows+1)


# missing_cust_ids = df.loc[df.isnull().any(axis=1),"customer_id"]
# print("Missing customer id's: ",missing_cust_ids)



import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans

df = pd.DataFrame({
    "Age": [20, 25, 30, 35, 40,60,70,90,100,120,240],
    "Salary": [20000, 25000, 30000, 35000, 40000,45000,50000,65000,112300,190120,1294920]
})


X = df.drop(['Age'],axis=1)
y = df['Age']

Scaled = StandardScaler()
X_scaled = Scaled.fit_transform(X=X)

maxScaled = MinMaxScaler()
X_max_scaled = maxScaled.fit_transform(X=X)


model = KMeans(
    max_iter=100,
    random_state=42,
    n_clusters=10
)
model.fit(X=X_scaled)


y_predict = model.predict(X_scaled)

print("y_predicitons of SS is : ",y_predict)



model = KMeans(
    max_iter=100,
    random_state=42,
    n_clusters=2
)
model.fit(X=X_max_scaled)


y_minmax_predict = model.predict(X_max_scaled)
print("y_predictions of MM is : ",y_minmax_predict)


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.DataFrame({
    "Age":[20,25,30,35,40,60,70,90,100,120,240],
    "Salary":[20000,25000,30000,35000,40000,
              45000,50000,65000,112300,190120,1294920]
})

X = df

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X=X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

labels = kmeans.fit_predict(X_scaled)

print(labels)

print("Kmeans_checkout predicitons: ",kmeans.fit_predict(df))
print("Kmeans STandard scaler predicyions: ",kmeans.fit_predict(X_scaled))
print("Kmeans of minmaxScaler ",kmeans.fit_predict(X_minmax))