import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# Create Images Folder
# ==========================================================

os.makedirs("images", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    iris = load_iris()

    X = iris.data
    y = iris.target

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["Species"] = [
        iris.target_names[i]
        for i in iris.target
    ]

    return iris, X, y, df

# ==========================================================
#  Dataset Information
# ==========================================================

def dataset_information(iris, X):

    print("=" * 70)
    print("IRIS FLOWER CLASSIFICATION")
    print("=" * 70)

    print("\nDataset Shape:", X.shape)

    print("\nFeature Names")

    for feature in iris.feature_names:
        print("-", feature)

    print("\nTarget Names")

    for target in iris.target_names:
        print("-", target)

# ==========================================================
# Class Distribution
# ==========================================================

def class_distribution(df):

    plt.figure(figsize=(6,4))

    sns.countplot(
        data=df,
        x="Species"
    )

    plt.title("Class Distribution")
    plt.xlabel("Species")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        "images/class_distribution.png",
        dpi=300
    )

    plt.show()

# ==========================================================
# Pair Plot
# ==========================================================

def pair_plot(df):

    pair = sns.pairplot(
        df,
        hue="Species"
    )

    pair.figure.savefig(
        "images/pairplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# ==========================================================
# Correlation Heatmap
# ==========================================================

def correlation_heatmap(df):

    plt.figure(figsize=(8,6))

    sns.heatmap(
        df.iloc[:, :-1].corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "images/correlation_heatmap.png",
        dpi=300
    )

    plt.show()
    
# ==========================================================
# Feature Distribution
# ==========================================================

def feature_distribution(df):

    df.iloc[:, :-1].hist(
        figsize=(10,8),
        bins=20
    )

    plt.tight_layout()

    plt.savefig(
        "images/feature_distribution.png",
        dpi=300
    )

    plt.show()

# ==========================================================
# Box Plot
# ==========================================================

def box_plot(df):

    plt.figure(figsize=(10,6))

    sns.boxplot(
        data=df.iloc[:, :-1]
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "images/boxplot.png",
        dpi=300
    )

    plt.show()

# ==========================================================
# Train-Test Split Dataset
# ==========================================================

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    return X_train, X_test, y_train, y_test

# ==========================================================
# Train Model
# ==========================================================

def train_model(X_train, y_train):
    """
    Train A Logistic Regression Model using the Training Data
    """

    model = LogisticRegression(
        max_iter=200
    )

    print("\nTraining Model...")

    model.fit(
        X_train,
        y_train
    )

    return model

# ==========================================================
# Make Predictions
# ==========================================================

def make_predictions(model, X_test, y_test, iris):

    predictions = model.predict(X_test)

    print("\nPredictions")

    for actual, predicted in zip(y_test, predictions):

        print(
            f"Actual: {iris.target_names[actual]:12}"
            f" Predicted: {iris.target_names[predicted]}"
        )

    return predictions

# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(y_test, predictions, iris):

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nAccuracy")
    print(f"{accuracy * 100:.2f}%")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names
        )
    )

    return cm

# ==========================================================
# Confusion Matrix Heatmap
# ==========================================================

def confusion_matrix_heatmap(cm, iris):

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(
        "images/confusion_matrix.png",
        dpi=300
    )

    plt.show()
    
# ==========================================================
# Save Predictions
# ==========================================================

def save_predictions(
    predictions,
    y_test,
    iris
):

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

    result.to_csv(
        "predictions.csv",
        index=False
    )

    print("\nPredictions saved.")


# ==========================================================
# Main Function
# ==========================================================

def main():

    iris, X, y, df = load_dataset()

    dataset_information(
        iris,
        X
    )

    class_distribution(df)

    pair_plot(df)

    correlation_heatmap(df)

    feature_distribution(df)

    box_plot(df)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    model = train_model(
        X_train,
        y_train
    )

    predictions = make_predictions(
        model,
        X_test,
        y_test,
        iris
    )

    cm = evaluate_model(
        y_test,
        predictions,
        iris
    )

    confusion_matrix_heatmap(
        cm,
        iris
    )

    save_predictions(
        predictions,
        y_test,
        iris
    )


if __name__ == "__main__":
    main()