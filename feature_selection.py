import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from log_code import setup_logging

logger = setup_logging("feature_selection")

import pickle
class FEATURE_SELECTION:

    @staticmethod
    def feature_selection(X_train, X_test, y_train):
        try:
            # ==========================================
            # Split numerical & categorical columns
            # ==========================================
            num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X_train.select_dtypes(include="object").columns

            X_train_num = X_train[num_cols].copy()
            X_test_num = X_test[num_cols].copy()

            X_train_cat = X_train[cat_cols].copy()
            X_test_cat = X_test[cat_cols].copy()

            logger.info(f"Numerical columns before FS: {num_cols.tolist()}")
            logger.info(f"Categorical columns untouched: {cat_cols.tolist()}")

            # ==========================================
            # 1️⃣ CONSTANT FEATURES
            # ==========================================
            vt_const = VarianceThreshold(threshold=0.0)
            vt_const.fit(X_train_num)

            removed_const = X_train_num.columns[~vt_const.get_support()]
            logger.info(f"Constant features removed: {removed_const.tolist()}")

            X_train_num = pd.DataFrame(
                vt_const.transform(X_train_num),
                columns=X_train_num.columns[vt_const.get_support()],
                index=X_train_num.index
            )
            X_test_num = pd.DataFrame(
                vt_const.transform(X_test_num),
                columns=X_test_num.columns[vt_const.get_support()],
                index=X_test_num.index
            )

            # ==========================================
            # 2️⃣ QUASI-CONSTANT FEATURES
            # ==========================================
            vt_quasi = VarianceThreshold(threshold=0.01)
            vt_quasi.fit(X_train_num)

            removed_quasi = X_train_num.columns[~vt_quasi.get_support()]
            logger.info(f"Quasi-constant features removed: {removed_quasi.tolist()}")

            X_train_num = pd.DataFrame(
                vt_quasi.transform(X_train_num),
                columns=X_train_num.columns[vt_quasi.get_support()],
                index=X_train_num.index
            )
            X_test_num = pd.DataFrame(
                vt_quasi.transform(X_test_num),
                columns=X_test_num.columns[vt_quasi.get_support()],
                index=X_test_num.index
            )

            # ==========================================
            # 3️⃣ CHI-SQUARE TEST
            # ==========================================
            X_train_chi = X_train_num.abs()
            X_test_chi = X_test_num.abs()

            chi_selector = SelectKBest(score_func=chi2, k="all")
            chi_selector.fit(X_train_chi, y_train)

            chi_features = X_train_num.columns[chi_selector.get_support()]
            logger.info(f"Features kept after Chi-Square: {chi_features.tolist()}")

            X_train_num = X_train_num[chi_features]
            X_test_num = X_test_num[chi_features]

            # ==========================================
            # 4️⃣ PEARSON CORRELATION TEST
            # ==========================================
            alpha = 0.05
            drop_features = []

            y_numeric = (
                pd.factorize(y_train)[0]
                if y_train.dtype == "object"
                else y_train.values
            )

            for col in X_train_num.columns:
                _, p_val = pearsonr(X_train_num[col], y_numeric)
                logger.info(f"{col} | p-value = {p_val:.6f}")

                if p_val > alpha:
                    drop_features.append(col)

            logger.info(f"Features removed by Pearson test: {drop_features}")

            X_train_num = X_train_num.drop(columns=drop_features)
            X_test_num = X_test_num.drop(columns=drop_features)

            # ==========================================
            # 🔗 Merge back categorical columns (UNCHANGED)
            # ==========================================
            X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
            X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

            logger.info(f"Final features after FS: {X_train_final.columns.tolist()}")
            logger.info(f"Final shape: {X_train_final.shape}")

            with open("feature_column.pkl", "wb") as f:
                pickle.dump(X_train_final.columns.tolist(), f)

            return X_train_final, X_test_final

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in feature_selection at line {error_line.tb_lineno}: {error_msg}"
            )
            return X_train, X_test