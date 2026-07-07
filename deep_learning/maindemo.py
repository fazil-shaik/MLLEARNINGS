import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
     

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout

from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris



df = load_iris()



print(df.feature_names)
print(df.target_names)



X = df.data
y = df.target

encoder = LabelEncoder()
y_int = encoder.fit_transform(y)

print(y_int)


X_train,X_test,y_train,y_test = train_test_split(X,y_int,test_size=0.2,random_state=42,stratify=y_int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
     

per = Perceptron(max_iter=1000,random_state=42)
per.fit(X_train_scaled,y_train)

y_pred_percep = per.predict(X_test_scaled)
     
accuracy = accuracy_score(y_test,y_pred_percep)

print("accuracy score we got is ",accuracy)


print("classification report: ",classification_report(y_test,y_pred_percep))


y_train_cat = to_categorical(y_train,num_classes = 3)
y_test_cat = to_categorical(y_test,num_classes = 3)


model = Sequential([
    Dense(16,input_dim=4,activation='relu'),
    Dense(8,activation='relu'),
    Dense(3,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train_scaled,y_train_cat,
                    epochs = 100,batch_size= 8, validation_split = 0.2,verbose = 1)


loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=1)
print(acc)


plt.figure(figsize = (10,4))
plt.plot(history.history['accuracy'],label = "train Acc")
plt.plot(history.history['val_accuracy'],label = "val Acc")


     