import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("creatine_data.csv")

print("Dataset Preview:")
print(data.head())

# -------------------------------
# 2. Features and Target
# -------------------------------
X = data.drop("response", axis=1)
y = data["response"]

# -------------------------------
# 3. Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 5. Model Training
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# -------------------------------
# 6. Prediction
# -------------------------------
y_pred = model.predict(X_test_scaled)

# -------------------------------
# 7. Evaluation
# -------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print("\nROC-AUC Score:", roc)

# -------------------------------
# 8. Feature Importance
# -------------------------------
importances = model.feature_importances_
features = X.columns

plt.figure()
plt.bar(features, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# -------------------------------
# 9. Predict for New User Input
# -------------------------------
print("\nEnter New User Data:")

age = float(input("Age: "))
weight = float(input("Weight (kg): "))
diet = int(input("Diet (0=Veg, 1=Non-Veg): "))
strength = float(input("Baseline Strength (kg): "))
sleep = float(input("Sleep Hours: "))
hydration = float(input("Hydration (liters/day): "))
intensity = int(input("Training Intensity (1-5): "))
bodyfat = float(input("Body Fat %: "))

new_data = np.array([[age, weight, diet, strength, sleep, hydration, intensity, bodyfat]])
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)[0]
probability = model.predict_proba(new_data_scaled)[0][1]

if prediction == 1:
    print("\nPrediction: Likely Responder to Creatine")
else:
    print("\nPrediction: Likely Non-Responder")

print("Response Probability:", round(probability, 2))