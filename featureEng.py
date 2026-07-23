import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


df = pd.DataFrame({
    "Age":[22,35,40,28,None],
    "Salary":[30000,50000,65000,40000,45000],
    "Gender":["Male","Female","Female","Male","Female"]
})

print(df)


#Handling missing values:

df["Age"] = df['Age'].fillna(df['Age'].mean())
print("After change of mean of null values: ")
print("="*20)
print(df)

#Encoding catogerial variables:
# For gender we have male and female machine understands only numbers not text
#using sklearn to encode the text to digits

encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])

print("After change of text to numbers : ")
print("="*20)
print(df)


#onehotEncoding better for non ordered cats:

# res = pd.get_dummies(df, columns=["Gender"])
# print(res)

#standardization:
# formulae:(x-mean)/std

#standard scaler

scaler = StandardScaler()
df[["Age","Salary"]] = scaler.fit_transform(df[["Age","Salary"]])

print(df)



#min max scaling:
