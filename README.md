# 📊 AI Powered Customer Churn Retention Prediction

An end-to-end machine learning project designed to predict customer churn and help telecom companies take proactive retention actions.  
The project covers the complete ML pipeline — from data visualization and preprocessing to model deployment using Flask.

---

## 🚀 Project Overview

Customer churn is a major challenge in the telecom industry. This project leverages machine learning to identify customers who are likely to leave the service by analyzing demographic details, service usage patterns, contract information, and billing data.

The system provides:
- Deep exploratory data analysis (EDA)
- Robust feature engineering and selection
- Balanced and scaled datasets
- A high-performing churn prediction model
- A Flask-based web interface for real-time predictions

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - Pandas, NumPy – Data manipulation & numerical computation
  - Matplotlib, Seaborn – Data visualization (EDA)
  - Scikit-learn – ML models & preprocessing
  - Imbalanced-learn (SMOTE) – Handling class imbalance
  - Flask – Model deployment & web interface
- **Model Persistence:** Pickle

---

## 📈 Exploratory Data Analysis (EDA)

Visualizations were created using **Matplotlib** and **Seaborn** to understand customer behavior and churn patterns.

### Key Insights:
- Majority of customers are retained, indicating **class imbalance**
- **Low-tenure customers churn more frequently**
- **Month-to-month contracts** have the highest churn
- **Senior citizens** show higher churn rates
- **Higher monthly charges** increase churn probability
- Gender has **minimal impact** on churn
- Fiber optic is the most widely adopted internet service

EDA helped identify trends, patterns, and potential outliers before model building.

---

## ⚙️ Feature Engineering

Feature engineering transformed raw data into meaningful inputs for modeling.

### Steps Performed:
- Identified missing values in `Total Charges`
- Converted object-type numerical columns correctly
- Applied **KNN Imputation** for accurate missing value handling
- Separated numerical and categorical features
- Performed variable transformation to reduce skewness
- Applied IQR-based outlier detection and clipping (Winsorization)

---

## 🧹 Handling Outliers

- Used **IQR-based detection**
- Compared **outlier clipping vs removal**
- Automatically selected the method that preserved maximum data
- Generated boxplots and CSV reports for transparency

This ensured data integrity while reducing the influence of extreme values.

---

## 🎯 Feature Selection

Feature selection was performed to improve model efficiency and prevent overfitting.

### Techniques Used:
- Constant & Quasi-Constant Feature Removal
- Variance Threshold
- Chi-Square Test (categorical features)
- Pearson Correlation Test (numerical features)

---

## 🔢 Categorical Encoding

Converted categorical variables into numerical form using:
- One-Hot Encoding
- Ordinal Encoding
- Target Encoding
- Binary Encoding
- Hashing Encoder (for high-cardinality features)

Correct encoding ensured model compatibility and avoided misleading relationships.

---

## ⚖️ Data Balancing

The dataset showed class imbalance, which was handled using **SMOTE (Synthetic Minority Over-sampling Technique)**.

### Why SMOTE?
- Generates synthetic minority samples
- Expands decision boundaries
- Improves generalization without duplicating data

---

## 📏 Feature Scaling

Feature scaling was applied to ensure uniform feature ranges.

### Scalers Evaluated:
- **Standard Scaler (Selected)** – Z-score normalization
- MinMax Scaler
- Robust Scaler
- MaxAbs Scaler

Standard Scaler was chosen for optimal model convergence.

---

## 🤖 Model Training & Evaluation

- Problem Type: **Binary Classification**
- Multiple ML algorithms were trained and evaluated
- Performance compared using **ROC Curve & AUC Score**
- **Logistic Regression** achieved the best AUC score

---

## 🏆 Best Model

- **Algorithm:** Logistic Regression
- **Metrics Used:**
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - ROC-AUC
- **Final Accuracy:** **75%**
- Model saved using Pickle for deployment

---

## 🌐 Deployment

A **Flask-based web application** allows users to:
- Input customer details
- Instantly receive churn predictions (Yes/No)

This bridges the gap between machine learning and real-world business decision-making.

---

## 📌 Conclusion

This project demonstrates a complete, production-ready machine learning pipeline for churn prediction.  
Key drivers of churn include:
- Tenure
- Contract type
- Monthly & total charges
- Service adoption
- Senior citizen status

The solution provides actionable insights that can significantly improve customer retention strategies.

---

## 🚀 Future Enhancements

- Deep Learning models (ANN, CNN, LSTM)
- Real-time churn prediction pipelines
- Interactive analytics dashboard
- Hybrid ML + Deep Learning models
- Advanced feature embedding for high-dimensional data

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to fork or contribute!
