from sklearn.linear_model import LogisticRegression


X = [[2], [6], [4], [1]]   # study hours
y = [0, 1, 1, 0]          # 0 = Fail, 1 = Pass

model = LogisticRegression()
model.fit(X, y)

print(f"\n Classification Single linear regression ")
print(model.predict_proba(X))
print(model.predict([[7]]))

# print(f"prediction percentage is {y_predict}")
# print(f"new Value predict is: {y_new_predict}")