import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# Load Dataset
# ==========================================================

iris = load_iris()

X = iris.data
y = iris.target


# ==========================================================
# Dataset Information
# ==========================================================

print("=" * 70)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 70)

print("\nDataset Shape:", X.shape)

print("\nFeature Names:")

for feature in iris.feature_names:
    print("-", feature)

print("\nTarget Names:")

for target in iris.target_names:
    print("-", target)


# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ==========================================================
# Build Model
# ==========================================================

model = LogisticRegression(max_iter=200)

print("\nTraining Model...")

model.fit(X_train, y_train)


# ==========================================================
# Predictions
# ==========================================================

predictions = model.predict(X_test)

print("\nPredictions")

for actual, predicted in zip(y_test, predictions):

    print(
        f"Actual: {iris.target_names[actual]:12}"
        f" Predicted: {iris.target_names[predicted]}"
    )


# ==========================================================
# Accuracy
# ==========================================================

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy")

print(f"{accuracy * 100:.2f}%")


# ==========================================================
# Confusion Matrix
# ==========================================================

print("\nConfusion Matrix")

cm = confusion_matrix(y_test, predictions)

print(cm)


# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)


# ==========================================================
# Save Predictions
# ==========================================================

result = pd.DataFrame({

    "Actual": [
        iris.target_names[i]
        for i in y_test
    ],

    "Predicted": [
        iris.target_names[i]
        for i in predictions
    ]

})

result.to_csv("predictions.csv", index=False)

print("\nPredictions saved as predictions.csv")