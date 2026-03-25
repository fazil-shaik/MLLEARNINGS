import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler




np.random.seed(42)
N = 500
GENRES = ['Pop', 'Rock', 'Classical', 'Hip-Hop', 'Jazz']

genre = np.random.choice(GENRES, N, p=[0.25, 0.20, 0.15, 0.25, 0.15])

# Audio features
tempo         = np.random.normal(120, 25, N)                          # BPM
valence       = np.clip(np.random.beta(2, 2, N), 0, 1)               # positivity
energy        = np.clip(np.random.beta(2, 2, N), 0, 1)               # intensity
acousticness  = np.clip(np.random.beta(3, 2, N), 0, 1)               # acoustic-ness
danceability  = np.clip(np.random.beta(2, 2, N), 0, 1)               # danceable
hours_per_day = np.clip(np.random.exponential(2, N), 0.5, 8)         # listening hrs

# Scientifically plausible: high valence = lower anxiety, high energy = higher anxiety
noise = np.random.normal(0, 0.8, N)
anxiety_score = np.clip(
    5.0
    - 2.5 * valence           # more positive music → less anxiety
    + 1.8 * energy            # intense music → more anxiety
    - 0.5 * acousticness      # acoustic → calming
    + 0.4 * hours_per_day     # more listening → slightly higher anxiety
    + 0.3 * (genre == 'Hip-Hop').astype(float)
    - 0.4 * (genre == 'Classical').astype(float)
    + noise,
    0, 10
)

dataframe = pd.DataFrame({
    "temp0":tempo,
    "valence":valence,
    "energy":energy,
    "acousticness":acousticness,
    "danceability":danceability,
    "hours_per_day":hours_per_day,
    "anxiety_score":anxiety_score
})

print(dataframe.head)
print(dataframe.shape)



X = dataframe.drop(['anxiety_score'],axis=1)
y = dataframe['anxiety_score']


#Train test split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection

LinearModel = LinearRegression()
LinearModel.fit(X_train,y_train)

#model predict and evaluation

y_linear_predict = LinearModel.predict(X_test)


#predicted values 
print("All predictions ",y_linear_predict)

#evaluations

print(f"evaluation is r2_score linear model {r2_score(y_test,y_linear_predict)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Then cross val
scores = cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=5, scoring='r2')

print(f"CV R² scores : {scores}")
print(f"Mean R²      : {scores.mean():.4f}")
print(f"Std          : {scores.std():.4f}")