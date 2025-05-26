import pandas as pd
import seaborn as sns

# Load penguins dataset
df = sns.load_dataset('penguins')
print(df.head())

#################
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier

# Load penguins dataset
df = sns.load_dataset('penguins')
print(df.head())

# Initialize default XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
print("XGBoost model initialized.")

# Note: This assumes dataset is already split (dependency on Person A's code)
# Placeholder for fitting (will be updated after merging)
print("Model fitting placeholder.")