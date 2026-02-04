# cat_to_num_best.py
import sys
import pandas as pd
import numpy as np

from log_code import setup_logging
logger = setup_logging("cat_to_num_best")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from category_encoders import (
    OrdinalEncoder,
    TargetEncoder,
    BinaryEncoder,
    HashingEncoder
)
import pickle
class CAT_TO_NUM:

    @staticmethod
    def apply_best(X_train, X_test, y_train):
        try:
            # ---------------- SPLIT COLUMNS ----------------
            cat_cols = X_train.select_dtypes(include="object").columns.tolist()
            num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

            if len(cat_cols) == 0:
                logger.info("No categorical columns found")
                return X_train, X_test

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train
            )

            results = {}

            # ---------------- ENCODERS (NO ONE-HOT) ----------------
            encoders = {
                "ordinal": OrdinalEncoder(cols=cat_cols),
                "target": TargetEncoder(cols=cat_cols),
                "binary": BinaryEncoder(cols=cat_cols),
                "hashing": HashingEncoder(cols=cat_cols, n_components=len(cat_cols))
            }

            for name, encoder in encoders.items():
                try:
                    X_tr_enc = encoder.fit_transform(X_tr[cat_cols], y_tr)
                    X_val_enc = encoder.transform(X_val[cat_cols])

                    # Combine with numeric
                    X_tr_final = pd.concat([X_tr[num_cols], X_tr_enc], axis=1)
                    X_val_final = pd.concat([X_val[num_cols], X_val_enc], axis=1)

                    # 🚨 Skip encoders that change feature count
                    if X_tr_final.shape[1] != X_train.shape[1]:
                        logger.info(f"{name} skipped (column mismatch)")
                        continue

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_tr_final, y_tr)
                    preds = model.predict_proba(X_val_final)[:, 1]

                    score = roc_auc_score(y_val, preds)
                    results[name] = score
                    logger.info(f"{name} ROC-AUC: {score:.4f}")

                except Exception as e:
                    logger.error(f"{name} failed: {e}")

            # ---------------- BEST METHOD ----------------
            best_method = max(results, key=results.get)
            logger.info(f"Best categorical encoder: {best_method}")

            # ---------------- APPLY BEST ----------------
            best_encoder = encoders[best_method]

            X_train_cat = best_encoder.fit_transform(X_train[cat_cols], y_train)
            X_test_cat = best_encoder.transform(X_test[cat_cols])

            X_train_final = pd.concat([X_train[num_cols], X_train_cat], axis=1)
            X_test_final = pd.concat([X_test[num_cols], X_test_cat], axis=1)

            logger.info(f"Final X_train shape: {X_train_final.shape}")
            logger.info(f"Final X_test shape: {X_test_final.shape}")

            with open("cat_encoder.pkl", "wb") as f:
                pickle.dump(best_encoder, f)

            return X_train_final, X_test_final

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error at line {error_line.tb_lineno}: {error_msg}")
            return X_train, X_test