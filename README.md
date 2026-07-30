# Iris Flower Classification

A beginner-friendly Machine Learning classification project built with **Logistic Regression** using the built-in **Iris dataset** from Scikit-learn. This project demonstrates the complete supervised learning workflow, from data exploration and visualization to model training, evaluation, and prediction.

## Features

* Load the Iris dataset from Scikit-learn
* Explore dataset structure, features, and target classes
* Visualize class distribution
* Generate pair plot for feature relationships
* Generate feature correlation heatmap
* Visualize feature distributions
* Detect potential outliers using box plots
* Split data into training and testing sets
* Train a Logistic Regression model
* Make predictions on test data
* Evaluate model performance using:

  * Accuracy Score
  * Confusion Matrix
  * Classification Report
* Visualize the confusion matrix as a heatmap
* Export predictions to a CSV file
* Automatically save all generated visualizations

## Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Project Structure

```text
Iris-Flower-Classification/
│
├── images/
│   ├── boxplot.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_distribution.png
│   └── pairplot.png
│
├── predictions.csv
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Machine Learning Workflow

```text
Load Dataset
      │
      ▼
Explore Dataset
      │
      ▼
Visualize Data
      │
      ▼
Train-Test Split
      │
      ▼
Train Logistic Regression Model
      │
      ▼
Make Predictions
      │
      ▼
Evaluate Model
      │
      ├── Accuracy Score
      ├── Confusion Matrix
      └── Classification Report
      │
      ▼
Visualize Results
      │
      ▼
Export Predictions
```

## Visualizations

The project automatically generates and saves the following visualizations:

* Class Distribution
* Pair Plot
* Correlation Heatmap
* Feature Distribution Histograms
* Box Plot
* Confusion Matrix Heatmap

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python main.py
```

## Learning Outcomes

Through this project, you will learn:

* Supervised Machine Learning
* Classification Problems
* Logistic Regression
* Dataset Exploration
* Data Visualization with Matplotlib and Seaborn
* Train-Test Split
* Model Training
* Making Predictions
* Model Evaluation
* Accuracy Score
* Confusion Matrix
* Classification Report
* Exporting Predictions

## Future Improvements

* Add user input for predicting a new Iris flower.
* Compare Logistic Regression with other classification algorithms.
* Save the trained model for future use.
* Add cross-validation and hyperparameter tuning.
