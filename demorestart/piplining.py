# #non linear models

# from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt

# X = [[1], [2], [3], [4], [5], [6], [7], [8]]
# y = [0, 0, 0, 0, 1, 1, 1, 1]

# model = LogisticRegression()
# model.fit(X=X,y=y)

# y_predict = model.predict([[2.5]])

# print(y_predict)

# y_probability = model.predict_proba(X)[:, 1]

# # plt.scatter(X,y=y,alpha=0.2,color='blue')
# # plt.plot(y_predict,color='red')
# # plt.xlabel('Actual hours invested in studies')
# # plt.ylabel('weather pass or fail')
# # plt.title("Prediction of simple pass/fail ")
# # plt.tight_layout()
# # plt.show()

# plt.scatter(X, y, alpha=0.6)

# plt.plot(X, y_probability)

# plt.xlabel("Hours invested in studies")
# plt.ylabel("Probability of passing")
# plt.title("Logistic Regression")

# plt.tight_layout()
# plt.show()



#DTR

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([
    30,
    35,
    42,
    50,
    58,
    65,
    80,
    95
])


model = DecisionTreeRegressor(max_depth=5)

model.fit(X,y)

prediction = model.predict([[4.5]])

print(prediction)

X_curve = np.linspace(1, 8, 200).reshape(-1, 1)

y_predict = model.predict(X_curve)

plt.scatter(X, y, alpha=0.7)

plt.plot(X_curve, y_predict)

plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.title("Decision Tree Regression")

plt.tight_layout()
plt.show()

