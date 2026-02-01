import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
sonar_data = pd.read_csv('data/sonar.csv', header=None)

# Split features and labels
X = sonar_data.drop(columns=[60])
Y = sonar_data[60]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

# Initialize model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, Y_train)

# Evaluate
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))
print(f"Training Accuracy: {train_acc}")
print(f"Test Accuracy: {test_acc}")

# Save model
joblib.dump(model, 'src/sonar_model.pkl')
print("Model saved to src/sonar_model.pkl")
