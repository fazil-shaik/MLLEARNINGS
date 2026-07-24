import pandas as pd
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
     "StudyHours": [1,2,3,4,5,6,7,8,9],
    "Result":     [0,0,0,0,1,1,1,1,1]
})

print(df.head())




# sns.scatterplot(
#     data=df,
#     x="StudyHours",
#     y="Result",
#     hue="Result",
#     s=120
# )

# plt.title("Study Hours vs Pass / Fail")
# plt.show()


X = df[['StudyHours']]
y = df['Result']


model = LogisticRegression()
model.fit(X,y)


# print(model.predict_proba(pd.DataFrame({"StudyHours": [7]})))

X_range = np.linspace(1, 9, 200).reshape(-1, 1)
probabilities = model.predict_proba(X_range)[:, 1]

sns.scatterplot(
    data=df,
    x="StudyHours",
    y="Result",
    hue="Result",
    s=120
)
new_value = 5.1
new_predicted = model.predict(pd.DataFrame({"StudyHours": [new_value]}))
plt.scatter(X,y,cmap='plasma',label='Actual value')
plt.plot(X_range, probabilities, color="black", linewidth=3)
plt.plot(new_value,new_predicted,color='purple',marker='*')
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression (Sigmoid Curve)")
plt.show()
