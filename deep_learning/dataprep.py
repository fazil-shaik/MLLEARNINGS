import pandas as pd

df = pd.read_csv("./customer.csv")

print(df.head())

print(df.tail())

print(df.shape)

print(df.info())

print("unique values count are : ",df.nunique().count())

print(df.describe())

print(df.isnull().sum())

new_dataframe = df.drop(
    columns=['customer_id','first_name','last_name']
)

print(new_dataframe.head())

print("Detect values: ",new_dataframe.isnull().sum())


missing_rows = new_dataframe[new_dataframe.isnull().any(axis=1)]

print('Missing rows are: ',missing_rows+1)


missing_cust_ids = df.loc[df.isnull().any(axis=1),"customer_id"]
print("Missing customer id's: ",missing_cust_ids)