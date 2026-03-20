from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n=100

hours = pd.date_range(start="2024-01-01", periods=n, freq="h")

# Base temperature with daily cycle + seasonal trend
hour_of_day = hours.hour
day_of_year = hours.day_of_year

temperature = (
    20
    + 10 * np.sin(2 * np.pi * day_of_year / 365)   # seasonal
    + 5  * np.sin(2 * np.pi * hour_of_day / 24)     # daily cycle
    + np.random.normal(0, 1.5, n)
)

# Humidity inversely related to temperature
humidity = np.clip(80 - 0.8 * temperature + np.random.normal(0, 5, n), 10, 100)

# Dew point
dew_point = temperature - ((100 - humidity) / 5)

# Pressure with slow drift
pressure = 1013 + np.cumsum(np.random.normal(0, 0.1, n))
pressure = np.clip(pressure, 980, 1040)

# Pressure change (tendency)
pressure_change = np.gradient(pressure)

# Wind speed
wind_speed = np.abs(np.random.normal(10, 5, n))

# Wind direction in degrees, encoded as sin/cos
wind_dir_deg = np.random.uniform(0, 360, n)
wind_dir_sin = np.sin(np.radians(wind_dir_deg))
wind_dir_cos = np.cos(np.radians(wind_dir_deg))

# Cloud cover
cloud_cover = np.clip(50 - pressure_change * 10 + np.random.normal(0, 15, n), 0, 100)

# Precipitation (sparse, higher when humidity high and pressure dropping)
precip_prob = 1 / (1 + np.exp(-(humidity - 75) * 0.1 - pressure_change * (-5)))
precipitation = np.where(np.random.rand(n) < precip_prob, np.random.exponential(2, n), 0)

# Lag features (t-1h, t-3h, t-6h)
def lag(arr, k):
    lagged = np.empty_like(arr)
    lagged[:k] = arr[0]
    lagged[k:] = arr[:-k]
    return lagged

temp_lag1  = lag(temperature, 1)
temp_lag3  = lag(temperature, 3)
temp_lag6  = lag(temperature, 6)
temp_lag12 = lag(temperature, 12)

humidity_lag3 = lag(humidity, 3)
humidity_lag6 = lag(humidity, 6)

pressure_lag3 = lag(pressure, 3)
pressure_lag6 = lag(pressure, 6)

# Rolling averages
def rolling_mean(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).mean().values

temp_roll3  = rolling_mean(temperature, 3)
temp_roll6  = rolling_mean(temperature, 6)
temp_roll12 = rolling_mean(temperature, 12)

# Temporal encodings
hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
month_sin = np.sin(2 * np.pi * hours.month / 12)
month_cos = np.cos(2 * np.pi * hours.month / 12)

# Interaction / derived features
temp_dew_spread = temperature - dew_point
heat_index = temperature + 0.33 * humidity - 4

# Target: temperature 24 hours ahead
target_temp_24h = lag(temperature, -24) if False else np.roll(temperature, -24)
target_temp_24h[-24:] = np.nan  # last 24 rows have no future target

df = pd.DataFrame({
    "timestamp": hours,
    "temperature": temperature,
    "humidity": humidity,
    "dew_point": dew_point,
    "pressure": pressure,
    "pressure_change": pressure_change,
    "wind_speed": wind_speed,
    "wind_dir_sin": wind_dir_sin,
    "wind_dir_cos": wind_dir_cos,
    "cloud_cover": cloud_cover,
    "precipitation": precipitation,
    "temp_lag1": temp_lag1,
    "temp_lag3": temp_lag3,
    "temp_lag6": temp_lag6,
    "temp_lag12": temp_lag12,
    "humidity_lag3": humidity_lag3,
    "humidity_lag6": humidity_lag6,
    "pressure_lag3": pressure_lag3,
    "pressure_lag6": pressure_lag6,
    "temp_roll3": temp_roll3,
    "temp_roll6": temp_roll6,
    "temp_roll12": temp_roll12,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "month_sin": month_sin,
    "month_cos": month_cos,
    "temp_dew_spread": temp_dew_spread,
    "heat_index": heat_index,
    "target_temp_24h": target_temp_24h,
})

df = df.dropna().reset_index(drop=True)

print(df.shape)
print(df.head())



X = df.drop(columns=['target_temp_24h', 'timestamp']).select_dtypes(include=[np.number])
y = df['target_temp_24h']


#model splitting 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#Model selection
LinearModel = LinearRegression()
LinearModel.fit(X_train,y_train)


#Prediction
y_linear_pred = LinearModel.predict(X_test)


#evaluate
print(f"MSE of Linear model {mean_squared_error(y_test,y_linear_pred)}")
print(f"r2_score of Linear model {r2_score(y_test,y_linear_pred)}")

#plotting 

features_to_plot = [
    'temperature', 'humidity', 'dew_point', 'pressure',
    'pressure_change', 'wind_speed', 'cloud_cover', 'precipitation',
    'temp_lag1', 'temp_lag3', 'temp_lag6', 'temp_lag12',
    'temp_roll3', 'temp_roll6', 'temp_roll12', 'heat_index'
]

n_cols = 4
n_rows = len(features_to_plot) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
axes = axes.flatten()


for i,feature in enumerate(features_to_plot):
    ax = axes[i]
    ax.plot(df['timestamp'], df[feature], linewidth=0.8, color='steelblue', alpha=0.7)
    ax.set_title(feature, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time', fontsize=8)
    ax.set_ylabel(feature, fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.4)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Weather Features Over Time', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1 - Actual vs Predicted (should be diagonal line)
axes[0].scatter(y_test, y_linear_pred, alpha=0.4, color='steelblue', s=10)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Actual vs Predicted')

# Plot 2 - Residuals (should be random around 0)
residuals = y_test - y_linear_pred
axes[1].scatter(y_linear_pred, residuals, alpha=0.4, color='tomato', s=10)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')

# Plot 3 - Residual distribution (should be bell curve)
axes[2].hist(residuals, bins=30, color='mediumseagreen', edgecolor='white')
axes[2].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[2].set_xlabel('Residual')
axes[2].set_ylabel('Count')
axes[2].set_title('Residual Distribution')

plt.tight_layout()
plt.show()

print(f"R² Score: {LinearModel.score(X_test, y_test):.4f}")



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Convert continuous target into 3 classes: Cold / Mild / Hot
bins   = [df['target_temp_24h'].min() - 1, 20, 27, df['target_temp_24h'].max() + 1]
labels = [0, 1, 2]  # 0=Cold, 1=Mild, 2=Hot
label_names = ['Cold', 'Mild', 'Hot']

y_class = pd.cut(df['target_temp_24h'], bins=bins, labels=labels).astype(int)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

LogisticModel = LogisticRegression(max_iter=1000, random_state=42)
LogisticModel.fit(X_train_c, y_train_c)

y_logistic_pred = LogisticModel.predict(X_test_c)

print("── Logistic Regression ──────────────────────────")
print(f"Accuracy:  {accuracy_score(y_test_c, y_logistic_pred):.4f}")
print(classification_report(y_test_c, y_logistic_pred, target_names=label_names))



print("── Model Comparison ─────────────────────────────")
print(f"{'Metric':<30} {'Linear Regression':>20} {'Logistic Regression':>20}")
print("-" * 72)
print(f"{'R² / Accuracy':<30} {r2_score(y_test, y_linear_pred):>20.4f} {accuracy_score(y_test_c, y_logistic_pred):>20.4f}")
print(f"{'MSE / N/A':<30} {mean_squared_error(y_test, y_linear_pred):>20.4f} {'N/A':>20}")
print(f"{'Task':<30} {'Regression':>20} {'Classification':>20}")
print(f"{'Target type':<30} {'Continuous':>20} {'Categorical (3 class)':>20}")



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1 — Confusion Matrix
cm = confusion_matrix(y_test_c, y_logistic_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names, ax=axes[0])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Plot 2 — Class probability distribution for each class
y_proba = LogisticModel.predict_proba(X_test_c)
colors = ['steelblue', 'tomato', 'mediumseagreen']
for idx, (cls, color) in enumerate(zip(label_names, colors)):
    axes[1].hist(y_proba[:, idx], bins=20, alpha=0.6, color=color, label=cls, edgecolor='white')
axes[1].set_title('Predicted Probabilities per Class', fontweight='bold')
axes[1].set_xlabel('Probability')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.4)

# Plot 3 — Side by side: Linear actual vs predicted | Logistic actual vs predicted class
axes[2].scatter(range(len(y_test_c)), y_test_c,  alpha=0.6, s=15, label='Actual',    color='steelblue')
axes[2].scatter(range(len(y_test_c)), y_logistic_pred, alpha=0.6, s=15, label='Predicted', color='tomato', marker='x')
axes[2].set_title('Logistic: Actual vs Predicted Class', fontweight='bold')
axes[2].set_xlabel('Sample Index')
axes[2].set_ylabel('Class (0=Cold, 1=Mild, 2=Hot)')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.4)

plt.suptitle('Logistic Regression Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()