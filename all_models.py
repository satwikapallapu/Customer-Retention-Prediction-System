import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import sys
import os

warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

# Optional: logging setup
from log_code import setup_logging
logger = setup_logging('All_Models')

# ---------------- MODEL FUNCTIONS ----------------
def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

def train_nb(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_lr(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_dt(X_train, y_train):
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, criterion='entropy')
    model.fit(X_train, y_train)
    return model

def train_ada(X_train, y_train):
    base = LogisticRegression()
    model = AdaBoostClassifier(estimator=base, n_estimators=50)
    model.fit(X_train, y_train)
    return model

def train_gb(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    return model

# ---------------- LOGISTIC TUNING ----------------
def tune_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [500, 1000]
    }
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    logger.info(f"Tuned Logistic Params: {grid.best_params_}")
    logger.info(f"Tuned Logistic CV ROC-AUC: {grid.best_score_}")
    return grid.best_estimator_

# ---------------- TRAIN & EVALUATE ALL MODELS ----------------
def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        "KNN": train_knn(X_train, y_train),
        "Naive Bayes": train_nb(X_train, y_train),
        "Logistic Regression": train_lr(X_train, y_train),
        "Decision Tree": train_dt(X_train, y_train),
        "Random Forest": train_rf(X_train, y_train),
        "AdaBoost": train_ada(X_train, y_train),
        "Gradient Boosting": train_gb(X_train, y_train),
        "XGBoost": train_xgb(X_train, y_train),
        "SVM": train_svm(X_train, y_train)
    }

    auc_scores = {}
    acc_scores = {}

    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], "k--")

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        auc_scores[name] = auc
        acc_scores[name] = acc

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (Acc={acc:.4f})')

        logger.info(f"{name} Test Accuracy: {acc}")
        logger.info(f"{name} ROC-AUC: {auc}")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - All Models")
    plt.legend(loc="lower right")
    plt.show()

    # ---------------- SELECT BEST MODEL ----------------
    best_model_name = max(auc_scores, key=auc_scores.get)
    best_auc = auc_scores[best_model_name]
    logger.info(f"BEST MODEL: {best_model_name} with ROC-AUC: {best_auc}")

    best_model = models[best_model_name]

    if best_model_name == "Logistic Regression":
        best_model = tune_logistic_regression(X_train, y_train)

    # ---------------- SAVE BEST MODEL ----------------
    with open("Churn_Prediction_Best_Model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"{best_model_name} saved successfully as Churn_Prediction_Best_Model.pkl")

    return models, best_model_name, best_model
