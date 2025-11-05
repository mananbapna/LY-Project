import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("processed/processed.csv", index_col=0)
print(f"Loaded dataset shape: {df.shape}")

# Split features and target
X = df.drop(columns=["churn"])
y = df["churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define and train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc'
)
model.fit(X_train, y_train)

# Plot feature importance
print("Plotting top 10 feature importances...")
xgb.plot_importance(model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Feature Importances (XGBoost)")
plt.tight_layout()

# Save plot
plt.savefig("outputs/feature_importance.png")
print("Saved feature importance plot to outputs/feature_importance.png")

