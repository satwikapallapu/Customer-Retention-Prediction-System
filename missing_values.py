# missing_values.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from log_code import setup_logging

logger = setup_logging("missing_values")

class MISSING_VALUES:

    @staticmethod
    def constant_imputer(df, fill_value=0):
        try:
            imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            return df_imputed
        except Exception:
            import sys
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in constant_imputer at line {error_line.tb_lineno}: {error_msg}")
            return df

    @staticmethod
    def mean_imputer(df):
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy="mean")
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            return df
        except Exception:
            import sys
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in mean_imputer at line {error_line.tb_lineno}: {error_msg}")
            return df

    @staticmethod
    def median_imputer(df):
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy="median")
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            return df
        except Exception:
            import sys
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in median_imputer at line {error_line.tb_lineno}: {error_msg}")
            return df

    @staticmethod
    def knn_imputer(df, n_neighbors=5):
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            return df
        except Exception:
            import sys
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in knn_imputer at line {error_line.tb_lineno}: {error_msg}")
            return df

    @staticmethod
    def find_best(X, y):
        """
        Finds the best missing value imputer based on numeric columns using Logistic Regression.
        Categorical columns are left as-is.
        """
        try:
            imputers = {
                "constant": MISSING_VALUES.constant_imputer,
                "mean": MISSING_VALUES.mean_imputer,
                "median": MISSING_VALUES.median_imputer,
                "knn": MISSING_VALUES.knn_imputer
            }

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.info("No numeric columns found for imputation. Skipping.")
                return "constant"

            best_score = 0
            best_method = None

            for name, func in imputers.items():
                X_copy = X.copy()
                X_copy = func(X_copy)  # Only numeric columns are imputed
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_copy[numeric_cols], y, test_size=0.2, random_state=42, stratify=y
                )
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_split, y_train_split)
                preds = model.predict_proba(X_val_split)[:, 1]
                score = roc_auc_score(y_val_split, preds)
                logger.info(f"{name} imputer ROC-AUC: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_method = name

            logger.info(f"Best imputer selected: {best_method} with ROC-AUC: {best_score:.4f}")
            return best_method

        except Exception:
            import sys
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in find_best at line {error_line.tb_lineno}: {error_msg}")
            return "constant"