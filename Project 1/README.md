# Project 1 — Decision Tree Classification (Breast Cancer Dataset)

## Overview
This project explores **decision tree classification** using scikit-learn on a medical dataset.
The objective is to understand how decision trees work, how model complexity affects performance, and how hyperparameters can be optimized.

The model is trained to classify tumors as **benign or malignant** based on diagnostic features.

## Dataset
- **Source:** Breast Cancer Wisconsin Dataset (loaded via scikit-learn)
- **Number of samples:** 569
- **Number of features:** 30
- **Classes:** Benign and Malignant

## Methods

### Data Preparation
- Dataset is loaded directly from scikit-learn
- Data is split into training and test sets using stratified sampling
- Features are used as provided (no manual feature engineering)

### Decision Tree Model
- Model: DecisionTreeClassifier
- Splitting criterion: Entropy
- Minimum samples required to split a node: 6
- Random state fixed for reproducibility

### Training & Evaluation
- Model is trained on the training set
- Predictions are generated on the test set
- Performance is evaluated using **classification accuracy**
- The trained decision tree is visualized to understand decision rules

## 📊 Results

### Decision Tree Performance
The decision tree classifier achieves strong accuracy on the test dataset.
The visualization of the tree provides insight into feature importance and decision paths used for classification.


![Decision Tree Visualization](figures/decision_tree.png)

### Accuracy vs Tree Depth
The model is trained using different maximum tree depths to study underfitting and overfitting.
- Shallow trees underfit the data
- Very deep trees overfit the training set
- An intermediate depth provides the best generalization performance

### Hyperparameter Optimization
GridSearchCV is used to find the best combination of:
- Maximum tree depth
- Minimum samples required to split a node

The optimized model improves stability and generalization.

## Summary
This project demonstrates how decision trees can be applied to real-world classification problems.
It highlights the importance of controlling model complexity and tuning hyperparameters to achieve better performance.

## How to Run / Reproduce Results

### Requirements
Python 3.8 or higher

Libraries: scikit-learn, matplotlib

### Setup

1. Clone the repository and enter Project 1:
```bash
git clone https://github.com/maazkhanzz/Machine-Learning.git
cd Machine-Learning/Project\ 1
