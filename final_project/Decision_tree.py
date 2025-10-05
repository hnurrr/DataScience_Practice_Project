import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# === LOAD DATASET ===
print("=== LOADING DATASET ===")
data = pd.read_csv("dava_sonuclari.csv")
print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\nFirst 5 rows:")
print(data.head())

# === DATA EXPLORATION ===
print("\n=== DATA EXPLORATION ===")
print("Data types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDescriptive statistics:")
print(data.describe())

# === DATA PREPARATION ===
print("\n=== DATA PREPARATION ===")
print("Encoding categorical variables...")
le = LabelEncoder()
data_ready = data.copy()
data_ready["Case Type"] = le.fit_transform(data_ready["Case Type"])
print(f"Encoded classes: {le.classes_}")

print("\nOutlier detection:")
numeric_cols = data_ready.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1 = data_ready[col].quantile(0.25)
    Q3 = data_ready[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    outliers = data_ready[(data_ready[col] < low) | (data_ready[col] > high)]
    print(f"{col}: {len(outliers)} outliers detected.")

print("\nTarget variable (Outcome) distribution:")
print(data_ready["Outcome"].value_counts())
print(f"Win rate: {(data_ready['Outcome'].sum() / len(data_ready)) * 100:.2f}%")

print("\nCorrelation analysis:")
if data_ready["Outcome"].nunique() > 1:
    corr = data_ready.corr()
    print("Top correlated features with Outcome:")
    print(corr["Outcome"].abs().sort_values(ascending=False).head(6))
else:
    print("⚠️ Only one class present, correlation not applicable.")

# === TRAIN-TEST SPLIT ===
print("\n=== DATA SPLITTING ===")
X = data_ready.drop("Outcome", axis=1)
y = data_ready["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data standardized.")

# === MODEL CREATION ===
print("\n=== MODEL TRAINING ===")
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt"
)

print("Training model...")
model.fit(X_train, y_train)
print("Model training completed.")

# === MODEL EVALUATION ===
print("\n=== MODEL EVALUATION ===")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

if len(np.unique(y_test)) > 1:
    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
else:
    print("⚠️ Only one class detected; some metrics unavailable.")
    train_prec = test_prec = train_rec = test_rec = train_f1 = test_f1 = 0

print("\n=== RESULTS ===")
print(f"{'Metric':<12} {'Train':<10} {'Test':<10}")
print(f"{'Accuracy':<12} {train_acc:<10.4f} {test_acc:<10.4f}")
print(f"{'Precision':<12} {train_prec:<10.4f} {test_prec:<10.4f}")
print(f"{'Recall':<12} {train_rec:<10.4f} {test_rec:<10.4f}")
print(f"{'F1-Score':<12} {train_f1:<10.4f} {test_f1:<10.4f}")

if train_acc - test_acc > 0.1:
    print("⚠️ Possible overfitting detected.")
else:
    print("✅ Model appears balanced.")

print("\nClassification Report:")
if len(np.unique(y_test)) > 1:
    print(classification_report(y_test, y_test_pred, target_names=["Lose", "Win"]))
else:
    print("Single class detected, report unavailable.")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

print("\nFeature Importance:")
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
print(feat_imp)

# === VISUALIZATION ===
print("\n=== VISUALIZATION ===")
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.barh(feat_imp["Feature"][:8], feat_imp["Importance"][:8])
plt.gca().invert_yaxis()
plt.title("Top Feature Importances")

plt.subplot(2, 2, 2)
if len(np.unique(y_test)) > 1:
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
else:
    plt.text(0.5, 0.5, "Single class", ha="center", va="center")
    plt.axis("off")

plt.subplot(2, 2, 3)
metrics = ["Accuracy", "Precision", "Recall", "F1"]
train_vals = [train_acc, train_prec, train_rec, train_f1]
test_vals = [test_acc, test_prec, test_rec, test_f1]
x = np.arange(len(metrics))
plt.bar(x - 0.2, train_vals, 0.4, label="Train")
plt.bar(x + 0.2, test_vals, 0.4, label="Test")
plt.xticks(x, metrics)
plt.legend()
plt.title("Model Performance")

plt.subplot(2, 2, 4)
plot_tree(model, max_depth=3, feature_names=X.columns, class_names=["Lose", "Win"], filled=True)
plt.title("Decision Tree (Top 3 Levels)")

plt.tight_layout()
plt.show()

# === INTERPRETATION ===
print("\n=== INTERPRETATION ===")
print(f"Test accuracy: {test_acc * 100:.2f}%")
if test_acc > 0.85:
    print("The model performs very well.")
elif test_acc > 0.7:
    print("The model performs reasonably; some improvement possible.")
else:
    print("The model is weak; further tuning is required.")

print("\nRecommendations:")
if len(np.unique(y)) == 1:
    print("- The dataset is imbalanced; both classes should be represented.")
else:
    print("- Collect more data samples.")
    print("- Try feature engineering techniques.")
    print("- Tune hyperparameters to improve performance.")

print("\n✅ Analysis complete.")


# In[ ]:




