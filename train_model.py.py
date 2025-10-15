import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic heart disease dataset
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(30, 80, n_samples),
    'sex': np.random.randint(0, 2, n_samples),  # 0: Female, 1: Male
    'cp': np.random.randint(0, 4, n_samples),  # Chest pain type
    'trestbps': np.random.randint(90, 200, n_samples),  # Resting blood pressure
    'chol': np.random.randint(120, 400, n_samples),  # Cholesterol
    'fbs': np.random.randint(0, 2, n_samples),  # Fasting blood sugar > 120 mg/dl
    'restecg': np.random.randint(0, 3, n_samples),  # Resting ECG results
    'thalach': np.random.randint(70, 200, n_samples),  # Max heart rate achieved
    'exang': np.random.randint(0, 2, n_samples),  # Exercise induced angina
    'oldpeak': np.random.uniform(0, 6, n_samples),  # ST depression
    'slope': np.random.randint(0, 3, n_samples),  # Slope of peak exercise ST
    'ca': np.random.randint(0, 4, n_samples),  # Number of major vessels
    'thal': np.random.randint(0, 4, n_samples)  # Thalassemia
}

df = pd.DataFrame(data)

# Create target variable with realistic patterns
df['target'] = 0
df.loc[(df['age'] > 55) & (df['chol'] > 240) & (df['trestbps'] > 140), 'target'] = 1
df.loc[(df['cp'] >= 2) & (df['exang'] == 1), 'target'] = 1
df.loc[(df['oldpeak'] > 2) & (df['ca'] >= 2), 'target'] = 1
df.loc[(df['sex'] == 1) & (df['age'] > 60) & (df['chol'] > 250), 'target'] = 1

# Add some randomness
noise = np.random.random(n_samples)
df.loc[noise > 0.85, 'target'] = 1 - df.loc[noise > 0.85, 'target']

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df['target'].value_counts())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model and scaler
with open('heart_attack_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved successfully!")
print("Files created: heart_attack_model.pkl, scaler.pkl")
