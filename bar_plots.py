import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('bar_plots')


def graphs(df):
    try:

        # CHURN DISTIBUTION
        a = (df['Churn'].value_counts(normalize=True).mul(100).plot(kind='bar', figsize=(7, 5)))
        plt.xlabel('churn')
        plt.ylabel('percentage')
        plt.title('churn Distribution')
        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.savefig('Churn.png')
        plt.show()

        #  CHURN VS TENURE
        df['tenure_bin'] = pd.cut(df['tenure'], bins=5)
        tenure_churn_pct = (pd.crosstab(df['tenure_bin'], df['Churn'], normalize='index') * 100)
        a = tenure_churn_pct.plot(kind='bar', figsize=(7, 5), color=['g', 'orange'])
        plt.title('Tenure vs Churn (%)')
        plt.xlabel('Tenure (Binned)')
        plt.ylabel('Percentage')
        plt.legend(title='Churn')
        for c in a.containers:
            a.bar_label(c, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('Tenure_and_churn.png')
        plt.show()

        # GENDER WISE CHURN DISTROBUTION
        churn_gender =df[df['Churn'] == 'Yes']['gender'].value_counts(normalize=True) * 100
        # Plot pie chart
        plt.figure(figsize=(5, 3))
        plt.pie(churn_gender, labels=churn_gender.index, autopct='%.1f%%', startangle=90)
        plt.title('Gender-wise Churn Distribution')
        plt.tight_layout()
        plt.savefig('Gender_churn_pie.png')
        plt.show()

        #  TOTALCHARGES AND CHURN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges_bin'] = pd.cut(df['TotalCharges'], bins=5)
        total_churn_pct = (pd.crosstab(df['TotalCharges_bin'], df['Churn'], normalize='index') * 100)
        a = total_churn_pct.plot(kind='bar', figsize=(7, 5), color=['g', 'orange'])
        plt.title('Total Charges vs Churn (%)')
        plt.xlabel('Total Charges (Binned)')
        plt.ylabel('Percentage')
        plt.legend(title='Churn')
        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('total_charges_and_churn.png')
        plt.show()

        # MONTHLYCHARGES VS CHURN
        df['MonthlyCharges_bin'] = pd.cut(df['MonthlyCharges'], bins=5)
        monthly_churn_pct = (pd.crosstab(df['MonthlyCharges_bin'], df['Churn'], normalize='index') * 100)
        a = monthly_churn_pct.plot(kind='bar', figsize=(7, 5), color=['g', 'orange'])
        plt.title('Monthly Charges vs Churn (%)')
        plt.xlabel('Monthly Charges (Binned)')
        plt.ylabel('Percentage')
        plt.legend(title='Churn')
        for c in a.containers:
            a.bar_label(c, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('Monthly_charges_vs_churn.png')
        plt.show()

        # CHURNED CUSTOMERS BY GENDER AND SENIORCITIZEN
        churn_df = df[df['Churn'] == 'Yes']
        churn_gender_senior = (pd.crosstab(churn_df['gender'], churn_df['SeniorCitizen'], normalize='index') * 100)
        churn_gender_senior.columns = ['Non_Senior', 'Senior']
        a = churn_gender_senior.plot(kind='bar', figsize=(7, 5))
        plt.xlabel('Gender')
        plt.ylabel('Percentage')
        plt.title('Churned customers by gender and Senior Citizen status')
        plt.legend(title='SeniorCitizen')
        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.savefig('churn_Gender_SeniorCitizen.png')
        plt.show()

        # INTERNET SERVICE USAGE BY GENDER
        gender_internet_pct = (pd.crosstab(df['gender'], df['InternetService'], normalize='index') * 100)
        a = gender_internet_pct.plot(kind='bar', figsize=(7, 5))
        plt.xlabel('Gender')
        plt.ylabel('Percentage')
        plt.title('Internet Service suage by gender')

        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('Gender_and_InternetService.png')
        plt.show()

        # PHONE SERVICE USAGE BY GENDER AND SENIORCITIZEN(CHURNED CUSTOMERS)
        churn_df = df[df['Churn'] == 'Yes']
        churn_df['Gender_Senior'] = (
                    churn_df['gender'] + '_' + churn_df['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'}))
        phone_gender_senior_pct = (
                    pd.crosstab(churn_df['Gender_Senior'], churn_df['PhoneService'], normalize='index') * 100)
        a = phone_gender_senior_pct.plot(kind='bar', figsize=(9, 5))
        plt.title('Phone service usage among churned customers by \nGender and Senior Citizen')
        plt.xlabel('Gender and SeniorCitizen')
        plt.ylabel('Percentage')
        plt.legend(title='Phone Service')
        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('Phone service_Gender_Senior_churn.png')
        plt.show()

        # MULTIPLE LINES USAGE BY GENDER AND SENIOR CITIZEN
        df['Gender_Senior'] = (df['gender'] + '_' + df['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'}))
        multiline_pct = (pd.crosstab(df['Gender_Senior'],df['MultipleLines'], normalize='index') * 100)
        a = multiline_pct.plot(kind='bar')
        plt.title('Multiple line usage by Gender and Senior Citizen')
        plt.xlabel('Gender & SeniorCitizen')
        plt.ylabel('Percentage')
        plt.legend(title='Multiple Lines')
        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('MultipleLines_gender_senior.png')
        plt.show()

        # MULTIPLE LINES BY SIM OPERATOR
        multilines_sim_pct = (pd.crosstab(df['SIM'], df['MultipleLines'], normalize='index') * 100)
        a = multilines_sim_pct.plot(kind='bar', figsize=(7, 5))
        plt.title('Multiple Lines usage by SIM operator')
        plt.xlabel('SIM operator')
        plt.ylabel('Percentage')
        plt.legend(title='Multiple Lines')

        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('Multiple_lines_by_SIM.png')
        plt.show()

        # TOTAL CUSTOMERS BY INTERNET SERVICE
        internet_total_pct = df['InternetService'].value_counts(normalize=True) * 100
        a = internet_total_pct.plot(kind='bar', figsize=(7, 5))
        plt.title('Cunsomers by Internet Service')
        plt.xlabel('Ineternet service')
        plt.ylabel('Percentage')
        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('Total_Internetservice.png')
        plt.show()

        # INTERNET SERVIVE USAGE BY SIM OPERATOR
        internet_sim_counts = pd.crosstab(df['InternetService'], df['SIM'])
        internet_sim_filtered = internet_sim_counts.loc[['DSL', 'Fiber optic', 'No']]
        a = internet_sim_filtered.plot(kind='bar', figsize=(7, 5))
        plt.title('Internet Service Usage by SIM operator')
        plt.xlabel('InternetService')
        plt.ylabel('Count')
        plt.legend(title='SIM')

        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('InternetSErvice_by_SIM.png')
        plt.show()

        # PAYMENT METHOD VS INTERNET SETVICE
        payment_internet_pct = (pd.crosstab(df['PaymentMethod'], df['InternetService'], normalize='index') * 100)
        a = payment_internet_pct.plot(kind='bar', figsize=(7, 5))
        plt.title('Internet Service Usage by Payment Method (%)')
        plt.xlabel('Payment Method')
        plt.ylabel('Percentage')
        plt.legend(title='Internet Service')

        for I in a.containers:
            a.bar_label(I, fmt='%.1f%%')

        plt.tight_layout()
        plt.savefig('payment vs internetservice.png')
        plt.show()

        # ONLINESECURITY,ONLINEBACKUP,DEVICEPROTECTION,TECHSUPPORT,STRAMINGTV,STREAMINGMOVIES
        service_cols = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]

        for i, col in enumerate(service_cols, 1):
            plt.subplot(2, 3, i)
            pct =df[col].value_counts(normalize=True) * 100
            a = pct.plot(kind='bar', figsize=(18, 10), color=['g', 'orange', 'red'])
            plt.title(f'{col} Usage (%)')
            plt.xlabel('')
            plt.ylabel('Percentage')
            for i in a.containers:
                a.bar_label(i, fmt='%.1f%%')

        plt.tight_layout()
        plt.savefig('ServiceColumns.png')
        plt.show()

        # CONTRACT VS PAYMENTS
        contract_churn_pct = (pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100)
        a = contract_churn_pct.plot(kind='bar')
        plt.title('Churn Percentage by Contract Type')
        plt.xlabel('Contract')
        plt.ylabel('Percentage')
        plt.legend(title='Churn')
        for i in a.containers:
            a.bar_label(i, fmt='%.1f%%')
        plt.tight_layout()
        plt.savefig('Churn_vs_Contract.png')
        plt.show()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')