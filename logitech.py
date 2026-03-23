from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets,linear_model,metrics
import matplotlib.pyplot as plt

DigitsData = datasets.load_digits()


X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")


X = DigitsData.data
y = DigitsData.target


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


Logimodel  = LogisticRegression(max_iter=1000,random_state=0)
Logimodel.fit(X_train,y_train)


y_pred = Logimodel.predict(X_test)

print(f"Logistic Regression model accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")


plt.figure(figsize=(10,8))
plt.scatter(X_train[: ,1],y_train,label='Actual test data')
plt.plot(y_test,y_pred,label='Predicted data',color='purple',linewidth=2)
plt.legend()
plt.show()




