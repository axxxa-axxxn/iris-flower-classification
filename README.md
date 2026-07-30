 # 🌸 Iris Flower Classification

A beginner-friendly **Machine Learning classification project** that predicts the species of an Iris flower using **Logistic Regression** and the built-in **Iris dataset** from Scikit-learn.

This project demonstrates the complete supervised machine learning workflow, including data exploration, visualization, model training, prediction, and performance evaluation.

---

## 📌 Project Overview

The Iris dataset is one of the most well-known datasets in machine learning and is commonly used to learn classification algorithms.

In this project, a Logistic Regression model is trained to classify Iris flowers into one of three species based on their physical measurements.

### Target Classes

- Iris Setosa
- Iris Versicolor
- Iris Virginica

---

## 📂 Project Structure

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

---

## 📊 Dataset Information

- **Dataset:** Iris Dataset (Scikit-learn)
- **Samples:** 150
- **Features:** 4
- **Classes:** 3

### Input Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target Variable

- Setosa
- Versicolor
- Virginica

---

## 🚀 Features

- Load the Iris dataset from Scikit-learn
- Explore dataset structure and statistics
- Check feature and target information
- Visualize class distribution
- Generate pair plots
- Create a correlation heatmap
- Visualize feature distributions
- Detect potential outliers using box plots
- Split the dataset into training and testing sets
- Train a Logistic Regression classifier
- Predict flower species
- Evaluate model performance
- Export predictions to a CSV file
- Automatically save all generated visualizations

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

These metrics help measure how well the classifier performs on unseen data.

---

## 📷 Generated Visualizations

The project automatically creates and saves the following visualizations:

- 📊 Class Distribution
- 🌸 Pair Plot
- 🔥 Correlation Heatmap
- 📉 Feature Distribution Histograms
- 📦 Box Plot
- ✅ Confusion Matrix Heatmap

---

## 🛠 Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🧠 Machine Learning Workflow

```text
Load Dataset
      │
      ▼
Explore Dataset
      │
      ▼
Data Visualization
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
Export Predictions & Visualizations
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Iris-Flower-Classification.git
```

Move into the project folder:

```bash
cd Iris-Flower-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## 🎯 Learning Outcomes

This project helped practice:

- Supervised Machine Learning
- Classification Problems
- Logistic Regression
- Exploratory Data Analysis (EDA)
- Data Visualization
- Train-Test Split
- Model Training
- Model Prediction
- Model Evaluation
- Confusion Matrix
- Accuracy Score
- Classification Report
- Exporting Predictions

---

## 🔮 Future Improvements

- Add user input for predicting a new Iris flower
- Compare Logistic Regression with Decision Tree, KNN, and Random Forest
- Save the trained model using Joblib
- Perform cross-validation
- Apply hyperparameter tuning
- Build a simple web application for predictions

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your work
5. Submit a Pull Request

---

## 📄 License

This project is open-source and available under the MIT License.