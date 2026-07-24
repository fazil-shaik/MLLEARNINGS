import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
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
# formulae-(x-min)/(max-min)

minmaxe = MinMaxScaler()
df[['Salary']] = minmaxe.fit_transform(df[['Salary']])
print("After Applying minmaxe Scaler : ")
print("="*20)
print(df)

#feature creation

# Suppose

# Height = 170 cm

# Weight = 70 kg

# Create

# BMI

# Formula

# BMI = weight/(height_in_meter)^2

df = pd.DataFrame({
    "Height":[170,160,180],
    "Weight":[70,60,90]
})

df["BMI"] = df["Weight"] / ((df["Height"]/100)**2)
print("="*20)
print(df)


# Text Feature Engineering
# Suppose
# "I love machine learning"
# Useful features
# Number of words
# Number of characters
# Average word length



df = pd.DataFrame({
    "Text":[
        "I love AI",
        "Machine Learning is awesome"
    ]
})

df["Words"] = df["Text"].apply(lambda x: len(x.split()))

df["Characters"] = df["Text"].str.len()

print(df)


#Log transformation

import numpy as np

df["Salary"] = np.log1p(df["Salary"])

print("="*20)

print(df)