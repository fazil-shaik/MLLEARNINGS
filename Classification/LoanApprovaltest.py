import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


X = np.array([[1], [2], [4], [6]])   # Income in lakhs
y = np.array([0, 0, 1, 1])           # Loan approval


model = LogisticRegression()
model.fit(X, y)


X_test = np.linspace(0, 7, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1


plt.figure()
plt.scatter(X, y, marker='o')        # Actual data points
plt.plot(X_test, y_prob)             # Sigmoid curve
plt.axhline(0.5, linestyle='--')     # Decision boundary
plt.axvline(
    x=(0 - model.intercept_[0]) / model.coef_[0][0],
    linestyle='--'
)

plt.xlabel("Monthly Income (Lakhs)")
plt.ylabel("Probability of Loan Approval")
plt.title("Logistic Regression - Loan Approval")
plt.show()


new_income = [[4.5]]
prob = model.predict_proba(new_income)[0][1]
prediction = model.predict(new_income)[0]

print(f"Approval probability for income 2.5 lakhs: {prob:.3f}")
print("Loan Approved" if prediction == 1 else "Loan Rejected")
