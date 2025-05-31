# 📉 Telecom Churn Prediction

This project uses machine learning to predict whether a telecom customer is likely to churn. By analyzing customer behavior and service attributes, telecom companies can proactively identify at-risk customers and implement retention strategies.

🔗 [Medium Blog](https://medium.com/@muhammedshabnaspa/predicting-customer-churn-in-telecom-a-machine-learning-approach-4d6dbf4e3fbf) 
📁 [View Notebook](telecom.ipynb)

---

## 🧠 Objective

- Analyze a telecom dataset to identify patterns in customer churn behavior.
- Build machine learning models to accurately predict churn.
- Use feature importance to understand which customer attributes influence churn the most.

---

## 📊 Dataset Overview

The dataset includes information such as:

- **Demographic features**: gender, senior citizen status, partner, dependents  
- **Service-related features**: phone service, internet service, contract type  
- **Account info**: tenure, billing method, monthly and total charges  
- **Target variable**: `Churn` (Yes/No)  

---

## 🛠 Tools & Technologies

| Tool              | Purpose                         |
|------------------|----------------------------------|
| Python            | Programming language             |
| Pandas & NumPy    | Data cleaning and manipulation   |
| Matplotlib & Seaborn | Data visualization             |
| Scikit-learn      | ML models and evaluation metrics |
| XGBoost           | Boosting algorithm               |
| Jupyter Notebook  | Interactive development          |

---

## 🔧 Workflow

### 1️⃣ Data Preprocessing

- Handled missing values in `TotalCharges` column.
- Converted categorical features using Label Encoding and One-Hot Encoding.
- Scaled numerical features using `StandardScaler`.

### 2️⃣ Exploratory Data Analysis (EDA)

- Visualized churn distribution across different contract types, payment methods, and tenure.
- Identified correlations using heatmaps.
- Noted that month-to-month contract customers churned more often.

### 3️⃣ Model Building

Split data into training and testing sets (80/20):

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Models implemented:

- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost Classifier

### 4️⃣ Model Evaluation

Metrics used:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

```python
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
```

✅ **Best Model:** XGBoost  
📈 AUC Score ~ 0.85

---

## 📌 Key Insights

- **Contract Type** is a strong churn predictor: Month-to-month customers have higher churn.
- **Tenure** is negatively correlated with churn: longer-tenured customers churn less.
- **Electronic Check** payment users are more likely to churn.

---

## 📊 Sample Results

| Model               | Accuracy | F1 Score | AUC Score |
|---------------------|----------|----------|-----------|
| Logistic Regression | 0.80     | 0.72     | 0.84      |
| Random Forest       | 0.82     | 0.75     | 0.86      |
| XGBoost             | **0.85** | **0.78** | **0.88**  |

---

## 🧭 Project Structure

```
📦 telecom/
├── telecom.ipynb                # Jupyter Notebook with all code
├── README.md                    # Project documentation
```

---

## 🚀 Future Improvements

- Use **GridSearchCV** for hyperparameter tuning  
- Apply **SMOTE** to handle class imbalance  
- Deploy the model via **Flask** or **Streamlit**  
- Schedule retraining on new customer data  

---

## 📎 Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Kaggle on Customer Churn](https://www.kaggle.com/search?q=churn)

---

## 👨‍💻 Author

**Muhammed Shabnas P A**  
📧 muhammedshabnaspa@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/muhammed-shabnas-pa/)  
🌐 [GitHub](https://github.com/Muhammed-Shabnas-PA)

---
