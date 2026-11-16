# ============================================
# ML TASK 2: SUPERVISED LEARNING
# Load and Explore Dataset
# ============================================

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer Wisconsin dataset
cancer = load_breast_cancer()

# Convert input features into a Pandas DataFrame
df = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

# Add target column
df["target"] = cancer.target

# Save the dataset as a CSV file
df.to_csv("data/breast_cancer.csv", index=False)

print("Dataset loaded successfully!")

# Display first five rows
print("\nFirst 5 Rows:")
print(df.head())

# Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display target class distribution
print("\nTarget Class Distribution:")
print(df["target"].value_counts())

# Display target class names
print("\nTarget Classes:")
print(cancer.target_names)

# ============================================
# Select Features and Target
# ============================================

# Select all input features
X = df.drop(columns=["target"])

# Select target variable
y = df["target"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ============================================
# Train/Test Split
# ============================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Features Shape:")
print(X_train.shape)

print("\nTesting Features Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

# Check class distribution after splitting

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())

# ============================================
# Missing Value Handling and Scaling
# ============================================

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Handle missing values using median imputation

imputer = SimpleImputer(strategy="median")

# Learn median values from training data
X_train_imputed = imputer.fit_transform(X_train)

# Apply the same learned values to testing data
X_test_imputed = imputer.transform(X_test)

print("\nMissing Value Handling Completed!")

print("Training Data Shape:", X_train_imputed.shape)
print("Testing Data Shape:", X_test_imputed.shape)

print("Missing Values After Imputation (Train):",
      np.isnan(X_train_imputed).sum())

print("Missing Values After Imputation (Test):",
      np.isnan(X_test_imputed).sum())

# ============================================
# Feature Standardization
# ============================================

scaler = StandardScaler()

# Learn mean and standard deviation from training data
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Apply the same transformation to testing data
X_test_scaled = scaler.transform(X_test_imputed)

print("\nFeature Scaling Completed!")

print("\nFirst 5 Original Training Samples:")
print(X_train.head())

print("\nFirst 5 Standardized Training Samples:")
print(X_train_scaled[:5])

print("\nMean of Scaled Training Features:")
print(np.mean(X_train_scaled, axis=0))

print("\nStandard Deviation of Scaled Training Features:")
print(np.std(X_train_scaled, axis=0))


# ============================================
# Train Logistic Regression
# ============================================

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train the model using scaled training data
logistic_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression Training Completed!")

# Make predictions on testing data
logistic_predictions = logistic_model.predict(X_test_scaled)

print("\nFirst 10 Predictions:")
print(logistic_predictions[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

# ============================================
# Logistic Regression Evaluation
# ============================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Calculate evaluation metrics
logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    pos_label=0,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    pos_label=0,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    pos_label=0,
    zero_division=0
)

print("\n========== LOGISTIC REGRESSION RESULTS ==========")

print(f"Accuracy:  {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall:    {logistic_recall:.4f}")
print(f"F1 Score:  {logistic_f1:.4f}")

print("\nConfusion Matrix (rows/columns ordered as 0, 1):")
print(confusion_matrix(y_test, logistic_predictions, labels=[0, 1]))

print("=================================================")

# ============================================
# Decision Tree Classification
# ============================================

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree model
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

# Train using imputed, unscaled training data
decision_tree_model.fit(X_train_imputed, y_train)

print("\nDecision Tree Training Completed!")

# Predict using testing data
decision_tree_predictions = decision_tree_model.predict(
    X_test_imputed
)

print("\nFirst 10 Decision Tree Predictions:")
print(decision_tree_predictions[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

# ============================================
# Decision Tree Evaluation
# ============================================

decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_predictions
)

# Check Decision Tree for possible overfitting

tree_train_accuracy = decision_tree_model.score(
    X_train_imputed,
    y_train
)

tree_test_accuracy = decision_tree_model.score(
    X_test_imputed,
    y_test
)

print("\n========== DECISION TREE OVERFITTING CHECK ==========")

print(f"Training Accuracy: {tree_train_accuracy:.4f}")
print(f"Testing Accuracy:  {tree_test_accuracy:.4f}")

print("====================================================")

decision_tree_precision = precision_score(
    y_test,
    decision_tree_predictions,
    pos_label=0,
    zero_division=0
)

decision_tree_recall = recall_score(
    y_test,
    decision_tree_predictions,
    pos_label=0,
    zero_division=0
)

decision_tree_f1 = f1_score(
    y_test,
    decision_tree_predictions,
    pos_label=0,
    zero_division=0
)

print("\n========== DECISION TREE RESULTS ==========")

print(f"Accuracy:  {decision_tree_accuracy:.4f}")
print(f"Precision: {decision_tree_precision:.4f}")
print(f"Recall:    {decision_tree_recall:.4f}")
print(f"F1 Score:  {decision_tree_f1:.4f}")

print("\nConfusion Matrix (rows/columns ordered as 0, 1):")
print(
    confusion_matrix(
        y_test,
        decision_tree_predictions,
        labels=[0, 1]
    )
)

print("==========================================")

# ============================================
# K-Nearest Neighbors Classification
# ============================================

from sklearn.neighbors import KNeighborsClassifier

# Create KNN model with 5 neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train KNN using standardized training data
knn_model.fit(X_train_scaled, y_train)

print("\nKNN Training Completed!")

# Make predictions using standardized testing data
knn_predictions = knn_model.predict(X_test_scaled)

print("\nFirst 10 KNN Predictions:")
print(knn_predictions[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

# ============================================
# KNN Model Evaluation
# ============================================

knn_accuracy = accuracy_score(
    y_test,
    knn_predictions
)

knn_precision = precision_score(
    y_test,
    knn_predictions,
    pos_label=0,
    zero_division=0
)

knn_recall = recall_score(
    y_test,
    knn_predictions,
    pos_label=0,
    zero_division=0
)

knn_f1 = f1_score(
    y_test,
    knn_predictions,
    pos_label=0,
    zero_division=0
)

print("\n========== KNN RESULTS ==========")

print(f"Accuracy:  {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall:    {knn_recall:.4f}")
print(f"F1 Score:  {knn_f1:.4f}")

print("\nConfusion Matrix (rows/columns ordered as 0, 1):")

print(
    confusion_matrix(
        y_test,
        knn_predictions,
        labels=[0, 1]
    )
)

print("=================================")

# ============================================
# Support Vector Machine (SVM)
# ============================================

from sklearn.svm import SVC

# Create SVM classifier
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

# Train SVM using standardized training data
svm_model.fit(X_train_scaled, y_train)

print("\nSVM Training Completed!")

# Make predictions on the testing dataset
svm_predictions = svm_model.predict(X_test_scaled)

print("\nFirst 10 SVM Predictions:")
print(svm_predictions[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

# ============================================
# SVM Model Evaluation
# ============================================

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

svm_precision = precision_score(
    y_test,
    svm_predictions,
    pos_label=0,
    zero_division=0
)

svm_recall = recall_score(
    y_test,
    svm_predictions,
    pos_label=0,
    zero_division=0
)

svm_f1 = f1_score(
    y_test,
    svm_predictions,
    pos_label=0,
    zero_division=0
)

print("\n========== SVM RESULTS ==========")

print(f"Accuracy:  {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall:    {svm_recall:.4f}")
print(f"F1 Score:  {svm_f1:.4f}")

print("\nConfusion Matrix (rows/columns ordered as 0, 1):")
print(
    confusion_matrix(
        y_test,
        svm_predictions,
        labels=[0, 1]
    )
)

print("=================================")

# ============================================
# Gaussian Naive Bayes Classification
# ============================================

from sklearn.naive_bayes import GaussianNB

# Create Gaussian Naive Bayes model
naive_bayes_model = GaussianNB()

# Train using imputed, unscaled training data
naive_bayes_model.fit(X_train_imputed, y_train)

print("\nGaussian Naive Bayes Training Completed!")

# Make predictions on testing data
naive_bayes_predictions = naive_bayes_model.predict(
    X_test_imputed
)

print("\nFirst 10 Naive Bayes Predictions:")
print(naive_bayes_predictions[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

# ============================================
# Gaussian Naive Bayes Evaluation
# ============================================

naive_bayes_accuracy = accuracy_score(
    y_test,
    naive_bayes_predictions
)

naive_bayes_precision = precision_score(
    y_test,
    naive_bayes_predictions,
    pos_label=0,
    zero_division=0
)

naive_bayes_recall = recall_score(
    y_test,
    naive_bayes_predictions,
    pos_label=0,
    zero_division=0
)

naive_bayes_f1 = f1_score(
    y_test,
    naive_bayes_predictions,
    pos_label=0,
    zero_division=0
)

print("\n========== GAUSSIAN NAIVE BAYES RESULTS ==========")

print(f"Accuracy:  {naive_bayes_accuracy:.4f}")
print(f"Precision: {naive_bayes_precision:.4f}")
print(f"Recall:    {naive_bayes_recall:.4f}")
print(f"F1 Score:  {naive_bayes_f1:.4f}")

print("\nConfusion Matrix (rows/columns ordered as 0, 1):")

print(
    confusion_matrix(
        y_test,
        naive_bayes_predictions,
        labels=[0, 1]
    )
)

print("==================================================")

# ============================================
# Compare All Five Algorithms
# ============================================

import os

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Create comparison table using actual model results
comparison_data = {
    "Algorithm": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "SVM",
        "Gaussian Naive Bayes"
    ],

    "Accuracy": [
        logistic_accuracy,
        decision_tree_accuracy,
        knn_accuracy,
        svm_accuracy,
        naive_bayes_accuracy
    ],

    "Precision": [
        logistic_precision,
        decision_tree_precision,
        knn_precision,
        svm_precision,
        naive_bayes_precision
    ],

    "Recall": [
        logistic_recall,
        decision_tree_recall,
        knn_recall,
        svm_recall,
        naive_bayes_recall
    ],

    "F1 Score": [
        logistic_f1,
        decision_tree_f1,
        knn_f1,
        svm_f1,
        naive_bayes_f1
    ]
}

# Convert dictionary to Pandas DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Rank models by F1 score, then accuracy
comparison_df = comparison_df.sort_values(
    by=["F1 Score", "Accuracy"],
    ascending=False
).reset_index(drop=True)

# Add ranking column
comparison_df.insert(
    0,
    "Rank",
    range(1, len(comparison_df) + 1)
)

# Display comparison table
print("\n========== MODEL COMPARISON ==========")
print(comparison_df.to_string(index=False))

# Save results as CSV
comparison_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")


# ============================================
# Model Performance Comparison Chart
# ============================================

import matplotlib.pyplot as plt
import numpy as np

# Model names and metric values
algorithms = comparison_df["Algorithm"]
accuracy_scores = comparison_df["Accuracy"]
f1_scores = comparison_df["F1 Score"]

# Create positions for bars
x = np.arange(len(algorithms))

# Width of each bar
width = 0.35

# Create figure
plt.figure(figsize=(12, 6))

# Plot Accuracy bars
plt.bar(
    x - width / 2,
    accuracy_scores,
    width,
    label="Accuracy"
)

# Plot F1 Score bars
plt.bar(
    x + width / 2,
    f1_scores,
    width,
    label="F1 Score"
)

# Add labels and title
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Score")
plt.title("Comparison of Five Supervised Learning Algorithms")

# Set algorithm names
plt.xticks(
    x,
    algorithms,
    rotation=20,
    ha="right"
)

# Set score range
plt.ylim(0, 1.05)

# Show legend
plt.legend()

# Add horizontal grid
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# Adjust layout
plt.tight_layout()

# Save graph
plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.close()

print("Model comparison chart saved successfully!")