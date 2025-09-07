# Titanic Survival Prediction

## Overview
This project analyzes the Titanic dataset to predict passenger survival. Using **data cleaning**, **feature engineering**, and **machine learning models** (Logistic Regression and Random Forest), we explore patterns that influenced survival and create a predictive model.

The notebook is fully **portfolio-ready**, showing step-by-step preprocessing, visualization, modeling, and evaluation.

---

## Dataset
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)  
- Features include passenger information like `Age`, `Sex`, `Pclass`, `SibSp`, `Parch`, `Fare`, and `Cabin`.  
- Target: `Survived` (0 = No, 1 = Yes)

---

## Project Workflow

### 1. Data Loading & Exploration
- Loaded dataset from GitHub using pandas.
- Checked for missing values and explored basic statistics.
- Visualized survival distribution, age distribution, and passenger class using seaborn.

### 2. Data Cleaning & Feature Engineering
- Filled missing values (`Embarked`, `Age`, `Fare`).
- Extracted `Title` from names (Mr, Mrs, Miss, Rare).
- Created `Deck` from `Cabin`.
- Added family features: `FamilySize` and `IsAlone`.
- Dropped unused columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
- One-hot encoding for categorical variables.
- Scaled numeric features (`Age`, `Fare`, `FamilySize`).

### 3. Model Training
- **Logistic Regression**: baseline (~82% accuracy).
- **Random Forest**: stronger model (~84% accuracy).  
- Evaluated using **accuracy**, **confusion matrix**, and **classification report**.

### 4. Model Tuning
- Tuned Random Forest using **GridSearchCV**.
- Hyperparameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- Achieved improved performance (~84–88% accuracy depending on train/test split).

### 5. Optional Extras
- Visualized **feature importance** to see which features influenced survival predictions most.
- Plotted **confusion matrix heatmap**.

---

## Results

- **Accuracy:** ~84% (Random Forest)  
- **Confusion Matrix:**  
[[92 13]
[16 58]]

- **Important Features:**  
`Sex_male`, `Pclass`, `Fare`, `Age`, `Title_Rare`, `IsAlone`  

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) or Jupyter.  
2. Install required libraries:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib

3. Run each cell step by step.
