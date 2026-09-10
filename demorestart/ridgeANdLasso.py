import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Pipelining import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet


np.random.seed(42)

n = 500

energy = np.random.uniform(0, 1, n)

danceability = (
    0.7 * energy +
    np.random.normal(0, 0.15, n)
)

loudness = (
    -5 + 4 * energy +
    np.random.normal(0, 0.5, n)
)

tempo = np.random.uniform(60, 180, n)

valence = np.random.uniform(0, 1, n)

acousticness = np.random.uniform(0, 1, n)

instrumentalness = np.random.uniform(0, 1, n)

speechiness = np.random.uniform(0, 0.5, n)

duration = np.random.uniform(120, 300, n)

# Actual relationship with popularity
popularity = (
    30
    + 25 * energy
    + 20 * danceability
    + 15 * valence
    - 10 * acousticness
    - 8 * instrumentalness
    + np.random.normal(0, 5, n)
)

df = pd.DataFrame({
    "energy": energy,
    "danceability": danceability,
    "loudness": loudness,
    "tempo": tempo,
    "valence": valence,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "speechiness": speechiness,
    "duration": duration,
    "popularity": popularity
})

print(df.head())


X = df.drop("popularity",axis=1)
y = df["popularity"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


linear = Pipeline([
    ('Scaler',StandardScaler()),
    ('model',LinearRegression())
])
linear.fit(X_train,y_train)

ridge = Pipeline([
    ('Scaler',StandardScaler()),
    ('model',Ridge(alpha=10))
])
ridge.fit(X_train,y_train)


lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(alpha=10))
])

lasso.fit(X_train, y_train)

models = {
    "Linear":linear,
    "Ridge":ridge,
    "Lasso":lasso
}

for name,model in models.items():
    print("\n", name)

    for feature, coef in zip(
        X.columns,
        model.named_steps["model"].coef_
    ):
        print(f"{feature:20} {coef:.3f}")



import matplotlib.pyplot as plt
import numpy as np

features = X.columns

linear_coef = linear.named_steps["model"].coef_
ridge_coef = ridge.named_steps["model"].coef_
lasso_coef = lasso.named_steps["model"].coef_

x = np.arange(len(features))
width = 0.25

plt.figure(figsize=(14, 6))

plt.bar(x - width, linear_coef, width, label="Linear")
plt.bar(x, ridge_coef, width, label="Ridge")
plt.bar(x + width, lasso_coef, width, label="Lasso")

plt.xticks(x, features, rotation=45, ha="right")
plt.ylabel("Coefficient")
plt.xlabel("Features")
plt.title("Linear vs Ridge vs Lasso Coefficients")
plt.legend()

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 6))

plt.bar(features, lasso_coef)

plt.axhline(0, linewidth=1)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Coefficient")
plt.title("Lasso Coefficients")

plt.tight_layout()
plt.show()