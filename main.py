import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load penguins dataset
df = sns.load_dataset('penguins')
print(df.head())

# Drop rows with missing values
df = df.dropna()

# Features and target
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split into training and testing sets.")

# Initialize XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
print("XGBoost model initialized.")

# Fit the model
model.fit(X_train, y_train)
print("Model trained successfully.")
