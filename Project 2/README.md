# Project 2 — Linear & Logistic Regression

## Overview
This project explores **regression and classification models** using scikit-learn.
It is divided into three parts:
- **Linear Regression** for predicting restaurant profit
- **Logistic Regression** for binary classification (hiring prediction)
- **Multi-class classification concepts** using One-vs-Rest and One-vs-One strategies

The project demonstrates how regression models can be trained, visualized, and evaluated using real datasets.

---

## Dataset

### Part 1 — Linear Regression
- **File:** `RegressionData.csv`
- **Features:**
  - `X`: population of a city
- **Target:**
  - `y`: profit/loss of a restaurant

### Part 2 — Logistic Regression
- **File:** `LogisticRegressionData.csv`
- **Features:**
  - `Score1`: first technical interview score
  - `Score2`: second technical interview score
- **Target:**
  - `y`: hiring decision (0 = rejected, 1 = hired)

---

## Methods

### Part 1 — Linear Regression
- Used **LinearRegression** from `sklearn.linear_model`
- Modeled the relationship:
  
  y = b₀ + b₁X

- Trained using **least squares optimization**
- Visualized:
  - Original data (scatter plot)
  - Best-fit regression line

### Part 2 — Logistic Regression (Binary Classification)
- Used **LogisticRegression** from `sklearn.linear_model`
- Predicted hiring decisions based on two exam scores
- Visualized:
  - Training data with class-based markers
  - Predicted labels to identify classification errors

### Part 3 — Multi-class Classification (Conceptual)
- Explained two strategies used with logistic regression:

#### One-vs-Rest (OvR)
- Train one classifier per class
- Each classifier distinguishes one class from all others
- Final prediction is the class with the highest probability

#### One-vs-One (OvO)
- Train a classifier for every pair of classes
- Each classifier votes for a class
- Final prediction is chosen by majority vote

---

## 📊 Results

### Linear Regression Results
- The learned regression line models the relationship between population size and restaurant profit
- Model parameters (bias and weight) are printed during execution
- The model predicts expected profit for a city with **18 habitants**

### Logistic Regression Results
- The classifier correctly predicts most training instances
- Some misclassifications are visible, highlighting real-world classification challenges
- Visualization clearly shows correct vs incorrect predictions

---

## Summary
This project demonstrates how:
- Linear regression can model continuous relationships
- Logistic regression can perform binary classification
- Multi-class problems can be handled using OvR and OvO strategies

It highlights the importance of visualization when interpreting regression and classification models.

---

## How to Run / Reproduce Results

### Requirements
- Python 3.8 or higher
- Libraries:
  - pandas
  - scikit-learn
  - matplotlib

### Dataset
Ensure the following files are placed inside the **Project 2** directory:
- `RegressionData.csv`
- `LogisticRegressionData.csv`

### Setup
1. Clone the repository and enter Project 2:
```bash
git clone https://github.com/maazkhanzz/Machine-Learning.git
cd Machine-Learning/Project\ 2
