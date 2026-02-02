import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("./salary_dtr.csv")

X = df[["ExperienceYears", "EducationLevel", "SkillScore", "Age"]]
y = df["Salary"]

new_value = [['8','2','45','25']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


dt = DecisionTreeRegressor(
    max_depth=4,
    min_samples_split=10,
    random_state=42
)


dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
y_new_value_predict = dt.predict(new_value)

dt_mse = mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)


print(f"new value prediction value is {y_new_value_predict}")

print("Decision Tree Regression")
print("MSE:", dt_mse)
print("R2 Score:", dt_r2)
print()
print("Linear Regression")
print("MSE:", lr_mse)
print("R2 Score:", lr_r2)


plt.figure(figsize=(22, 12))
plot_tree(
    dt,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

