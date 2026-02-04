"""
In this file we are going to load the data and other Ml pipeline techniques
which are needed
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from log_code import setup_logging
from bar_plots import graphs
from sklearn.model_selection import train_test_split
from missing_values import MISSING_VALUES
from sklearn.metrics import accuracy_score, f1_score
from variable_transformation import VARIABLE_TRANSFORMATION
from feature_selection import FEATURE_SELECTION

import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('main_1')

from cat_to_num import CAT_TO_NUM
from data_balancing import DATA_BALANCING
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler
)
from sklearn.impute import SimpleImputer
from all_models import train_all_models



class TELE_CUSTOMER:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(self.df.head())
            logger.info(self.df.shape)
            logger.info('Data loaded successfully')
            logger.info(f'Null values:\n{self.df.isnull().sum()}')

            # Splitting data
            self.y = self.df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
            self.X = self.df.drop(['Churn', 'customerID'], axis=1)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            logger.info(f'X_train_shape: {self.X_train.shape}')
            logger.info(f'X_test_shape: {self.X_test.shape}')
            logger.info(f'y_train_shape: {self.y_train.shape}')
            logger.info(f'y_test_shape: {self.y_test.shape}')
            logger.info(f'y_train:{self.y_train.head()}')
            logger.info(f'X_train:{self.X_train.head()}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error at line {error_line.tb_lineno}: {error_msg}')

    def visualization(self):
        try:
            graphs(self.df)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error at line {error_line.tb_lineno}: {error_msg}')

    # inside TELE_CUSTOMER class
    def missing_val(self):
        try:
            best_method = MISSING_VALUES.find_best(self.X_train, self.y_train)
            logger.info(f"Best missing value method detected: {best_method}")

            if best_method == "constant":
                self.X_train = MISSING_VALUES.constant_imputer(self.X_train)
                self.X_test = MISSING_VALUES.constant_imputer(self.X_test)
            elif best_method == "mean":
                self.X_train = MISSING_VALUES.mean_imputer(self.X_train)
                self.X_test = MISSING_VALUES.mean_imputer(self.X_test)
            elif best_method == "median":
                self.X_train = MISSING_VALUES.median_imputer(self.X_train)
                self.X_test = MISSING_VALUES.median_imputer(self.X_test)
            elif best_method == "knn":
                self.X_train = MISSING_VALUES.knn_imputer(self.X_train)
                self.X_test = MISSING_VALUES.knn_imputer(self.X_test)

            logger.info(f'X_train:{self.X_train.sample(10)}')
            logger.info(f'X_train:{self.X_train.isnull().sum()}')

            logger.info("Missing value imputation applied successfully.")

        except Exception:
            import sys
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in missing_val at line {error_line.tb_lineno}: {error_msg}")


    def var_out(self):
        try:
            # Apply best numeric outlier handling
            self.X_train = VARIABLE_TRANSFORMATION.apply_best(self.X_train)
            self.X_test = VARIABLE_TRANSFORMATION.apply_best(self.X_test)
            logger.info(f'X_train:{self.X_train.sample(10)}')
            logger.info(f'X_train:{self.X_train.isnull().sum()}')
            logger.info("Variable transformation (outliers handling) completed.")

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in var_out at line {error_line.tb_lineno}: {error_msg}")

    def feature_selection(self):
        try:
            self.X_train, self.X_test = FEATURE_SELECTION.feature_selection(
                self.X_train, self.X_test, self.y_train
            )
            logger.info(f"X_train_shape: {self.X_train.shape}")
            logger.info(f"X_test_shape: {self.X_test.shape}")
            logger.info(f'X_train:{self.X_train.sample(10)}')


        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in feature_selection at line {error_line.tb_lineno}: {error_msg}"
            )

    def cat_to_numeric(self):
        try:
            self.X_train, self.X_test = CAT_TO_NUM.apply_best(
                self.X_train,
                self.X_test,
                self.y_train)

            logger.info(f"X_train_shape: {self.X_train.shape}")
            logger.info(f"X_test_shape: {self.X_test.shape}")
            logger.info(f'X_train:{self.X_train.sample(10)}')
            logger.info("Categorical encoding applied successfully")
        except Exception:
            import sys
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in cat_to_numeric at line {error_line.tb_lineno}: {error_msg}")

    def balance_data(self):
        try:
            self.X_train, self.y_train, self.X_test = DATA_BALANCING.data_balancing(
                self.X_train,
                self.y_train,
                self.X_test,
                force_balance=False
            )

            logger.info("===== DATA BALANCING COMPLETED =====")
            logger.info(f"Balanced X_train shape: {self.X_train.shape}")
            logger.info(f"Balanced y distribution:\n{self.y_train.value_counts()}")

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in balance_data at line {error_line.tb_lineno}: {error_msg}"
            )

    def feature_scaling(self):
        try:
            logger.info("Selecting Best Feature Scaling Technique")

            X_train = self.X_train.copy()
            X_test = self.X_test.copy()

            numeric_cols = X_train.columns.tolist()

            # -----------------------------
            # Handle missing values FIRST
            # -----------------------------
            imputer = SimpleImputer(strategy="median")

            X_train = pd.DataFrame(
                imputer.fit_transform(X_train),
                columns=numeric_cols,
                index=X_train.index
            )

            X_test = pd.DataFrame(
                imputer.transform(X_test),
                columns=numeric_cols,
                index=X_test.index
            )

            logger.info("Missing values handled using Median Imputation")

            # -----------------------------
            # Scoring function
            # -----------------------------
            def score_df(df):
                return df.skew().abs().mean() + df.kurtosis().abs().mean()

            scalers = {
                "STANDARD": StandardScaler(),
                "MINMAX": MinMaxScaler(),
                "ROBUST": RobustScaler(),
                "MAXABS": MaxAbsScaler()
            }

            best_score = np.inf
            best_scaler = None
            best_name = None

            for name, scaler in scalers.items():
                temp = X_train.copy()
                temp[numeric_cols] = scaler.fit_transform(temp[numeric_cols])

                score = score_df(temp[numeric_cols])
                logger.info(f"{name} scaler score = {score:.4f}")

                if score < best_score:
                    best_score = score
                    best_scaler = scaler
                    best_name = name

            logger.info(f"BEST SCALER SELECTED = {best_name}")

            # -----------------------------
            # Apply best scaler
            # -----------------------------
            self.X_train[numeric_cols] = best_scaler.fit_transform(X_train[numeric_cols])
            self.X_test[numeric_cols] = best_scaler.transform(X_test[numeric_cols])

            with open("best_scaler.pkl", "wb") as f:
                pickle.dump(best_scaler, f)

            train_all_models(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test
            )

            logger.info("Feature Scaling Completed Successfully")
            logger.info(f"X_train shape after scaling: {self.X_train.shape}")

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in feature_scaling at line {error_line.tb_lineno}: {error_msg}"
            )


if __name__ == '__main__':
    try:
        obj = TELE_CUSTOMER(f'C:\\Users\\satwi\\Downloads\\churn_project\\Churn_data.csv')
        #obj.visualization()
        obj.missing_val()
        obj.var_out()
        obj.feature_selection()
        obj.cat_to_numeric()
        obj.balance_data()
        obj.feature_scaling()




    except Exception:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f'Error at line {error_line.tb_lineno}: {error_msg}')