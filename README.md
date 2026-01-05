# Customer Churn Prediction System

## 📌 Problem Statement
Customer churn directly affects business revenue. This project predicts whether a customer will leave a service using Machine Learning.

## 📊 Dataset
Telecom customer dataset containing demographic, account, and usage information.

Target variable:
- Churn (1 = Yes, 0 = No)

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn

## 🧠 ML Models Used
- Logistic Regression
- Random Forest Classifier

## 🔄 Workflow
1. Data Loading
2. Data Encoding (Label Encoding)
3. Feature Scaling
4. Model Training
5. Evaluation using Accuracy, Precision, Recall, F1-score

## 📈 Results
Random Forest performed better due to its ability to handle non-linearity and feature interactions.

## 🚀 Future Enhancements
- Handle class imbalance using SMOTE
- Hyperparameter tuning
- Model deployment using Flask

## 🧪 How to Run
'''bash
pip install -r requirements.txt
python churn_model.py
