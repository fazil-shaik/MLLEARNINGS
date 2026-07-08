# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
     

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import Perceptron
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import tensorflow as tf

# from tensorflow.keras.models import Sequential

# from tensorflow.keras.layers import Dense

# from tensorflow.keras.layers import Dropout

# from tensorflow.keras.utils import to_categorical
# from sklearn.datasets import load_iris



# df = load_iris()



# print(df.feature_names)
# print(df.target_names)



# X = df.data
# y = df.target

# encoder = LabelEncoder()
# y_int = encoder.fit_transform(y)

# print(y_int)


# X_train,X_test,y_train,y_test = train_test_split(X,y_int,test_size=0.2,random_state=42,stratify=y_int)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)
     

# per = Perceptron(max_iter=1000,random_state=42)
# per.fit(X_train_scaled,y_train)

# y_pred_percep = per.predict(X_test_scaled)
     
# accuracy = accuracy_score(y_test,y_pred_percep)

# print("accuracy score we got is ",accuracy)


# print("classification report: ",classification_report(y_test,y_pred_percep))


# y_train_cat = to_categorical(y_train,num_classes = 3)
# y_test_cat = to_categorical(y_test,num_classes = 3)


# model = Sequential([
#     Dense(16,input_dim=4,activation='relu'),
#     Dense(8,activation='relu'),
#     Dense(3,activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# history = model.fit(X_train_scaled,y_train_cat,
#                     epochs = 100,batch_size= 8, validation_split = 0.2,verbose = 1)


# loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=1)
# print(acc)


# plt.figure(figsize = (10,4))
# plt.plot(history.history['accuracy'],label = "train Acc")
# plt.plot(history.history['val_accuracy'],label = "val Acc")



# sample = np.array([[5.1, 3.5, 1.4, 1.2]])
# prediction = model.predict(sample)
# predicted_class = np.argmax(prediction)

# print("\nPredicted Probabilities (Softmax Output):", prediction)
# print("Predicted Class:", df.target_names[predicted_class])


#cnn with minst dataset 


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron    # Used for simple linear classification tasks.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential     # Sequential lets you build a neural network layer-by-layer in Keras.

from tensorflow.keras.layers import Dense     #Dense makes the final predictions
from tensorflow.keras.layers import Conv2D     # Conv2D extracts features
from tensorflow.keras.layers import Flatten    # Flatten reshapes them

from tensorflow.keras.layers import MaxPooling2D     # MaxPooling2D reduces size
from tensorflow.keras.layers import Dropout          # Dropout prevents overfitting

from tensorflow.keras.utils import to_categorical

from sklearn.datasets import load_digits



df = pd.read_csv('./mnist_test.csv')
df_test = pd.read_csv("mnist_test.csv")


print(df.head())

print(df.describe())


res = df.isnull().sum()

print("not Null values are ",res)


X_train = df.drop("label", axis=1).values
y_train = df["label"].values
X_test = df_test.drop("label", axis=1).values
y_test = df_test["label"].values

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0 



X_train_img = X_train.reshape(-1, 28, 28)
X_test_img = X_test.reshape(-1, 28, 28)
     


y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
     

perceptron = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(10, activation="softmax")
])
     

perceptron.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
     

history_percp = perceptron.fit(X_train_img, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test_img, y_test_cat), verbose=1)
     
acc_percp = perceptron.evaluate(X_test_img, y_test_cat, verbose=0)[1]

print("="*50)

print("ANN: ",acc_percp)

ann = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history_ann = ann.fit(X_train_img, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test_img, y_test_cat), verbose=1)
     

acc_ann = ann.evaluate(X_test_img, y_test_cat, verbose=0)[1]

print("ANN accuracy score is: ",acc_ann)

X_train_cnn = X_train.reshape(-1, 28, 28,1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
     

cnn = Sequential([
    Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])
     

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
     

history_cnn = cnn.fit(X_train_cnn, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test_cnn, y_test_cat), verbose=1)
     
acc_cnn = cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]


print("CNN accuracy socre: ",acc_cnn)

def plot_training(history, title):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label="Train")
    plt.plot(history.history['val_accuracy'], label="Val")
    plt.title(f"{title} Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Val")
    plt.title(f"{title} Loss")
    plt.legend()
    plt.show()


plot_training(history_percp, "Perceptron")



plt.figure(figsize=(10,6))
plt.plot(history_percp.history['val_accuracy'], label="Perceptron")
plt.plot(history_ann.history['val_accuracy'], label="ANN")
plt.plot(history_cnn.history['val_accuracy'], label="CNN")
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Val Accuracy")
plt.legend()
plt.show()
     



def show_side_by_side(models, model_names, X, X_cnn, y_true, n=5):
    idxs = np.random.choice(len(X), n, replace=False)
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(idxs):
        plt.subplot(2, n, i+1)
        plt.imshow(X[idx].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.title(f"True: {y_true[idx]}")
        preds = [np.argmax(model.predict(X_cnn[idx].reshape(1, 28, 28, 1) if name == "CNN" else X[idx].reshape(1, 28, 28)))
                 for model, name in zip(models, model_names)]
        plt.subplot(2, n, n+i+1)
        plt.axis("off")
        plt.title("\n".join(f"{n}: {p}" for n, p in zip(model_names, preds)))
    plt.tight_layout()
    plt.show()
     

show_side_by_side([perceptron, ann, cnn], ["Perceptron", "ANN", "CNN"], X_test_img, X_test_cnn, y_test, 5)


final_accs = [acc_percp*100, acc_ann*100, acc_cnn*100]
models = ["Perceptron", "ANN", "CNN"]

plt.figure(figsize=(8,6))
bars = plt.bar(models, final_accs, color=['#ff9999','#66b3ff','#99ff99'])
plt.title("Final Test Accuracy Comparison")
plt.ylabel("Accuracy (%)")
for bar, acc in zip(bars, final_accs):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()-1, f"{acc:.2f}%",
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.ylim(80, 100)
plt.show()