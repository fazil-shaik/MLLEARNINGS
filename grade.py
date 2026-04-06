import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


np.random.seed(42)

n_samples = 500

data = {
    'temp_volatility': np.random.uniform(0.5, 4.0, n_samples),  # °C, daily swing
    'dissolved_oxygen': np.random.uniform(2, 8, n_samples),      # mg/L
    'ph': np.random.uniform(6.5, 8.5, n_samples),                # pH scale
    'ammonia_ppm': np.random.uniform(0.1, 3.0, n_samples),       # mg/L (toxin)
    'stocking_density': np.random.uniform(50, 300, n_samples),   # fish/m³
    'recent_rainfall_mm': np.random.uniform(0, 150, n_samples),  # mm in past 7 days
    'days_since_exchange': np.random.uniform(5, 45, n_samples),  # days
    'feed_quality': np.random.randint(1, 6, n_samples),          # 1-5 scale
}

df = pd.DataFrame(data)

outbreak_probability = np.zeros(n_samples)

stress_1 = (df['ammonia_ppm'] > 1.5) & (df['temp_volatility'] > 2.5)
outbreak_probability += stress_1 * 0.4

stress_2 = (df['dissolved_oxygen'] < 4) & (df['stocking_density'] > 200)
outbreak_probability += stress_2 * 0.35

stress_3 = ((df['ph'] < 7.2) | (df['ph'] > 8.3)) & (df['recent_rainfall_mm'] > 80)
outbreak_probability += stress_3 * 0.3

stress_4 = (df['days_since_exchange'] > 30) & (df['feed_quality'] < 3)
outbreak_probability += stress_4 * 0.25

noise = np.random.normal(0, 0.15, n_samples)
outbreak_probability = np.clip(outbreak_probability + noise, 0, 1)

df['outbreak'] = (outbreak_probability > 0.5).astype(int)

print("Class distribution:")
print(df['outbreak'].value_counts())
print(f"\nOutbreak rate: {df['outbreak'].mean():.1%}")

print("\nFirst 5 samples:")
print(df.head())

X = df.drop('outbreak', axis=1)
y = df['outbreak']

print(f"\nShape: X={X.shape}, y={y.shape}")
print(f"Features: {list(X.columns)}")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

GradientModel = GradientBoostingClassifier(n_estimators=1500,learning_rate=0.05,max_depth=3,random_state=42)
GradientModel.fit(X_train,y_train)

y_prediction = GradientModel.predict(X_test)


print("\nClassification Report:")
print(classification_report(y_test,y_prediction))

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_prediction))

print(f"Accuracy: {accuracy_score(y_test,y_prediction):.2%}")

plt.style.use('dark_background')
fig,axes  = plt.subplots(1,4,figsize=(20,5))
axes[0].scatter(df['temp_volatility'], df['outbreak'], alpha=0.5)
axes[0].plot(df['temp_volatility'], df['outbreak'], 'o', alpha=0.5)
axes[0].set_xlabel('Temperature Volatility (°C)')
axes[0].set_ylabel('Disease Outbreak')
axes[0].set_title('Temp Volatility vs Outbreak')
axes[1].scatter(df['dissolved_oxygen'], df['outbreak'], alpha=0.5)
axes[1].plot(df['dissolved_oxygen'], df['outbreak'], 'x', alpha=0.5)
axes[1].set_xlabel('Dissolved Oxygen (mg/L)')
axes[1].set_title('Dissolved Oxygen vs Outbreak')
axes[2].scatter(df['ammonia_ppm'], df['outbreak'], alpha=0.5)
axes[2].plot(df['ammonia_ppm'], df['outbreak'], 's', alpha=0.5)
axes[2].set_xlabel('Ammonia (ppm)')
axes[2].set_title('Ammonia vs Outbreak')
#actual bs predicted
axes[3].scatter(y_test, y_prediction, alpha=0.5)
axes[3].plot([0, 1], [0, 1], 'r--')  # Diagonal line for reference
axes[3].set_xlabel('Actual Outbreak')
axes[3].set_ylabel('Predicted Outbreak')
axes[3].set_title('Actual vs Predicted Outbreak')
plt.tight_layout()
plt.show()
