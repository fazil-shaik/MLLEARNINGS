import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

dataframe = pd.read_csv('Bank_data.csv')

# print(dataframe.head())

#data cleaning 

new_dataframe = dataframe.drop(['Employment_Length'], axis=1)

print(new_dataframe.head())

checking_null_values = new_dataframe.isnull().sum()

if checking_null_values.any():
    print("Null values found in the dataset. Please handle them before proceeding.",checking_null_values)
else:
    print("No null values found in the dataset. Proceeding with data preprocessing.")

# Data preprocessing
X = new_dataframe.drop(['Loan_Default'], axis=1)
y = new_dataframe['Loan_Default']

categorical_cols = ['Education', 'Loan_Purpose', 'Home_Ownership', 'Marital_Status']

# Option A: Label Encoding
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Option B: One-Hot Encoding (uncomment if preferred)
# X = pd.get_dummies(X, columns=categorical_cols)

# Encode target
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Convert to numpy arrays
X_array = X.values
y_array = y

print("X shape:", X_array.shape)
print("y shape:", y_array.shape)
print("X dtype:", X_array.dtype)
print("y dtype:", y_array.dtype)

    # Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

    # Dimensionality reduction using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

    # Clustering using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_pca)

    # Predicting clusters for the test set
y_pred_clusters = kmeans.predict(X_test_pca)

print("Clustering completed. Predicted clusters for the test set:")
print(y_pred_clusters)


#neural network model


class LoanDefaultNN(nn.Module):
    def __init__(self, input_size):
        super(LoanDefaultNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Reshape

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # Reshape

input_size = X_train_tensor.shape[1]
model = LoanDefaultNN(input_size)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 100

for epoch in range(epochs):

    model.train()

    outputs = model(X_train_tensor)

    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch+1) % 10 == 0:

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}"
        )

#moodel evaluation
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    y_pred = (y_pred_probs >= 0.5).float()


#accuracy calculation
accuracy = accuracy_score(
    y_test_tensor,
y_pred
)

print(f"Accuracy : {accuracy*100:.2f}%")


#confusion matrix and classification report
conf_matrix = confusion_matrix(y_test_tensor, y_pred)
class_report = classification_report(y_test_tensor, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


#predicting new data
customer = [[
    32,
    85000,
    2,
    740,
    1,
    250000,
    0,
    0.32,
    35000,
    150000,
    1,
    1
]]

customer = scaler.transform(customer)

customer = torch.FloatTensor(customer)

model.eval()

with torch.no_grad():

    probability = model(customer)

print("Default Probability:", probability.item())

if probability.item() >= 0.5:
    print("High Risk")
else:
    print("Low Risk")



#save model 
torch.save(
    model.state_dict(),
    "loan_default_model.pth"
)


model = LoanDefaultNN(input_size)

model.load_state_dict(
    torch.load("loan_default_model.pth")
)

model.eval()