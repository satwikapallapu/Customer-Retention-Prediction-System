# transformation.py
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

logger = logging.getLogger("variable_outliers")

class VARIABLE_TRANSFORMATION:

    @staticmethod
    def check_outliers(df, threshold=1.5):
        """Check numeric outliers using IQR; returns dict {col: outlier count}"""
        outlier_count = {}
        try:
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outliers = df[(df[col] < lower) | (df[col] > upper)]
                outlier_count[col] = outliers.shape[0]
            return outlier_count
        except Exception as e:
            logger.error(f"Error in check_outliers: {e}")
            return {}

    @staticmethod
    def handle_outliers(df, strategy="clip", threshold=1.5):
        """Handle numeric outliers using 'clip' or 'remove'"""
        df = df.copy()
        try:
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                if strategy == "clip":
                    df[col] = df[col].clip(lower, upper)
                elif strategy == "remove":
                    df = df[(df[col] >= lower) & (df[col] <= upper)]
                else:
                    raise ValueError("Invalid strategy: choose 'clip' or 'remove'")
            return df
        except Exception as e:
            logger.error(f"Error in handle_outliers: {e}")
            return df

    @staticmethod
    def save_outlier_plots(df, folder_name="outlier_images", prefix="before"):
        """Save boxplots of numeric columns"""
        try:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            num_cols = df.select_dtypes(include=["int64", "float64"]).columns
            for col in num_cols:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=df[col], color="skyblue")
                plt.title(f"{prefix.capitalize()} Outliers: {col}")
                plt.tight_layout()
                plt.savefig(os.path.join(folder_name, f"{prefix}_{col}.png"))
                plt.close()
        except Exception as e:
            logger.error(f"Error in save_outlier_plots: {e}")

    @staticmethod
    def apply_best(df):
        """Automatically apply the best numeric outlier handling strategy"""
        try:
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns

            # Save original numeric outliers and images
            original_outliers = VARIABLE_TRANSFORMATION.check_outliers(df)
            pd.DataFrame.from_dict(original_outliers, orient="index", columns=["outliers_before"]).to_csv("outliers_before.csv")
            VARIABLE_TRANSFORMATION.save_outlier_plots(df, prefix="before")
            logger.info(f"Outliers before handling: {original_outliers}")

            # Try both strategies
            best_df = df.copy()
            strategies = ["clip", "remove"]
            min_outliers = float("inf")
            best_strategy = None

            for strat in strategies:
                temp_df = VARIABLE_TRANSFORMATION.handle_outliers(df, strategy=strat)
                outlier_count = VARIABLE_TRANSFORMATION.check_outliers(temp_df)
                total_outliers = sum(outlier_count.values())

                if total_outliers < min_outliers:
                    min_outliers = total_outliers
                    best_df = temp_df
                    best_strategy = strat

            # Save numeric outliers after handling
            final_outliers = VARIABLE_TRANSFORMATION.check_outliers(best_df)
            pd.DataFrame.from_dict(final_outliers, orient="index", columns=["outliers_after"]).to_csv("outliers_after.csv")
            VARIABLE_TRANSFORMATION.save_outlier_plots(best_df, prefix="after")
            logger.info(f"Best numeric strategy: {best_strategy}")
            logger.info(f"Outliers after handling: {final_outliers}")

            return best_df

        except Exception as e:
            logger.error(f"Error in apply_best: {e}")
            return df