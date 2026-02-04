<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Powered Churn Retention Prediction</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            background-color: #f4f6f9;
            margin: 0;
            padding: 40px;
            color: #333;
        }
        .container {
            max-width: 1100px;
            margin: auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.08);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            text-align: center;
        }
        ul {
            line-height: 1.7;
        }
        code {
            background: #eef1f5;
            padding: 3px 6px;
            border-radius: 4px;
        }
        .badge {
            display: inline-block;
            background: #3498db;
            color: #fff;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 13px;
            margin-right: 6px;
        }
        .section {
            margin-top: 35px;
        }
    </style>
</head>
<body>

<div class="container">

    <h1>📊 AI Powered Customer Churn Retention Prediction</h1>

    <p style="text-align:center;">
        <span class="badge">Machine Learning</span>
        <span class="badge">EDA</span>
        <span class="badge">Feature Engineering</span>
        <span class="badge">Flask App</span>
        <span class="badge">Logistic Regression</span>
    </p>

    <div class="section">
        <h2>📌 Project Overview</h2>
        <p>
            This project focuses on predicting customer churn using an end-to-end
            machine learning pipeline. The goal is to help telecom companies identify
            customers who are likely to leave and take proactive retention actions.
        </p>
        <p>
            The project includes data visualization, preprocessing, feature engineering,
            model training, evaluation, and deployment using a Flask-based web interface.
        </p>
    </div>

    <div class="section">
        <h2>🛠️ Tech Stack</h2>
        <ul>
            <li><b>Python</b></li>
            <li><b>Pandas & NumPy</b> – Data preprocessing and numerical operations</li>
            <li><b>Matplotlib & Seaborn</b> – Data visualization (EDA)</li>
            <li><b>Scikit-learn</b> – ML models and preprocessing</li>
            <li><b>Imbalanced-learn (SMOTE)</b> – Data balancing</li>
            <li><b>Flask</b> – Web application deployment</li>
        </ul>
    </div>

    <div class="section">
        <h2>📈 Exploratory Data Analysis (EDA)</h2>
        <ul>
            <li>Churn distribution analysis revealed class imbalance</li>
            <li>Tenure vs churn showed higher churn among new customers</li>
            <li>Monthly & total charges strongly influence churn behavior</li>
            <li>Senior citizens have higher churn rates</li>
            <li>Gender has minimal impact on churn</li>
            <li>Month-to-month contracts show the highest churn</li>
        </ul>
    </div>

    <div class="section">
        <h2>⚙️ Feature Engineering</h2>
        <ul>
            <li>Handled missing values in <b>Total Charges</b> using KNN Imputation</li>
            <li>Separated numerical and categorical features</li>
            <li>Applied IQR-based outlier detection and clipping (Winsorization)</li>
            <li>Performed variable transformation to reduce skewness</li>
        </ul>
    </div>

    <div class="section">
        <h2>🎯 Feature Selection</h2>
        <ul>
            <li>Removed constant and quasi-constant features</li>
            <li>Used Variance Threshold for low-variance removal</li>
            <li>Applied Chi-Square test for categorical feature relevance</li>
            <li>Used Pearson correlation for numerical feature filtering</li>
        </ul>
    </div>

    <div class="section">
        <h2>🔢 Categorical Encoding</h2>
        <ul>
            <li>One-Hot Encoding for nominal variables</li>
            <li>Ordinal Encoding for ordered categories</li>
            <li>Explored Target, Binary, and Hashing Encoders</li>
        </ul>
    </div>

    <div class="section">
        <h2>⚖️ Data Balancing</h2>
        <p>
            SMOTE (Synthetic Minority Over-sampling Technique) was used to address
            class imbalance by generating synthetic minority samples, improving
            model generalization.
        </p>
    </div>

    <div class="section">
        <h2>📏 Feature Scaling</h2>
        <ul>
            <li>Standard Scaler (Z-score normalization) – Primary choice</li>
            <li>MinMax, Robust, and MaxAbs Scalers evaluated</li>
        </ul>
    </div>

    <div class="section">
        <h2>🤖 Model Training & Evaluation</h2>
        <ul>
            <li>Binary classification problem</li>
            <li>Multiple ML models evaluated</li>
            <li>ROC Curve & AUC used for comparison</li>
            <li><b>Logistic Regression</b> selected as best-performing model</li>
        </ul>
    </div>

    <div class="section">
        <h2>🏆 Best Model Results</h2>
        <ul>
            <li>Algorithm: Logistic Regression</li>
            <li>Evaluation Metrics: Accuracy, Confusion Matrix, Classification Report</li>
            <li>Final Accuracy: <b>75%</b></li>
            <li>Model saved using Pickle for deployment</li>
        </ul>
    </div>

    <div class="section">
        <h2>🌐 Deployment</h2>
        <p>
            A Flask-based web application allows users to input customer details
            and receive instant churn predictions, bridging ML insights with
            real-world business decisions.
        </p>
    </div>

    <div class="section">
        <h2>📌 Conclusion</h2>
        <p>
            This project demonstrates a complete machine learning workflow for
            churn prediction, highlighting the importance of data preprocessing,
            feature engineering, and model evaluation. The insights gained can
            help businesses improve customer retention strategies effectively.
        </p>
    </div>

    <div class="section">
        <h2>🚀 Future Enhancements</h2>
        <ul>
            <li>Integration of Deep Learning models (ANN, CNN, LSTM)</li>
            <li>Real-time churn prediction pipelines</li>
            <li>Interactive visual analytics dashboard</li>
            <li>Hybrid ML + Deep Learning approaches</li>
            <li>Advanced feature embedding techniques</li>
        </ul>
    </div>

</div>

</body>
</html>
