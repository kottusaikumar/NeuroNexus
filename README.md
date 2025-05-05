# 🚢 Titanic Survival Prediction - Machine Learning Project

This project applies various **supervised classification algorithms** to predict passenger survival from the Titanic disaster using the classic Titanic dataset. The model is built, evaluated, and compared using key performance metrics including accuracy, f1-score, and confusion matrix.

---

## 📌 Business Goal

The goal of this project is to:
- Build a predictive model that classifies whether a passenger survived or not.
- Understand key factors influencing survival.
- Compare model performances using precision, recall, and f1-score due to imbalanced data.
- Interpret feature importance and model behavior.

---

## 📂 Dataset Description

The dataset contains demographic and personal information of Titanic passengers. Below are the main columns used:

| Column      | Description |
|-------------|-------------|
| `Survived`  | Target variable (0 = No, 1 = Yes) |
| `Pclass`    | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Sex`       | Gender (male/female) |
| `Age`       | Age in years |
| `SibSp`     | # of siblings/spouses aboard |
| `Parch`     | # of parents/children aboard |
| `Fare`      | Passenger fare |
| `Embarked`  | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 🧠 Algorithms Used

This project compares multiple classification models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree
- Random Forest
- XGBoost

---

## 📊 Model Evaluation Metrics

- **F1-Score (Micro Average)**: Balances precision and recall; useful for imbalanced datasets.
- **Confusion Matrix**: Visual performance evaluation.
- **Cross-Validation Score**: Validates model generalization capability.
- **Bias-Variance Analysis**: Checks underfitting/overfitting using train vs test score comparison.

---

## 📈 Results

All models achieved high training and test performance. However:

- **XGBoost**, **Random Forest**, and **SVM** provided the most stable results.
- **Decision Tree** and **Naive Bayes** were simpler but still performed well.
- Model choice may depend on the priority between **interpretability** (Logistic, Decision Tree) vs **accuracy** (XGBoost, SVM).

---

## 🛠️ Libraries Used

Install all required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost simple-colors

