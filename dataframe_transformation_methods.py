import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from dataframe_transformation_methods import *

imp = Imputer()
imp_categorical = Imputer(strategy='most_frequent')


def map_and_impute(dataframe):
    name_contract_status_map = {'Cash loans': 0, 'Revolving loans': 1}
    dataframe.NAME_CONTRACT_TYPE = dataframe.NAME_CONTRACT_TYPE.map(name_contract_status_map)
    name_code_gender_map = {'F': 0, 'M': 1, 'XNA': 2}
    dataframe.CODE_GENDER = dataframe.CODE_GENDER.map(name_code_gender_map)
    name_own_car_map = {'N': 0, 'Y': 1}
    dataframe.FLAG_OWN_CAR = dataframe.FLAG_OWN_CAR.map(name_own_car_map)
    name_own_realty_map = {'N': 0, 'Y': 1}
    dataframe.FLAG_OWN_REALTY = dataframe.FLAG_OWN_REALTY.map(name_own_realty_map)
    name_income_type_map = {'Working': 1, 'Commercial associate': 2, 'Pensioner': 3, 'State servant':4 , 'Unemployed':5, 'Student':6, 'Businessman':7, 'Maternity leave':8}
    dataframe.NAME_INCOME_TYPE = dataframe.NAME_INCOME_TYPE.map(name_income_type_map)
    name_education_type_map = {'Secondary / secondary special': 1, 'Higher education': 2, 'Incomplete higher': 3, 'Lower secondary': 4, 'Academic degree': 5}
    dataframe.NAME_EDUCATION_TYPE = dataframe.NAME_EDUCATION_TYPE.map(name_education_type_map)
    name_family_status = {'Married': 1, 'Single / not married': 2, 'Civil marriage': 3,
                               'Separated': 4, 'Widow': 5, 'Unknown': 6}
    dataframe.NAME_FAMILY_STATUS = dataframe.NAME_FAMILY_STATUS.map(name_family_status)
    return dataframe


def transform_dataframe_bureau(dataframe):
    df_b = pd.read_csv('bureau.csv')
    df_b_filtered = df_b.groupby(['SK_ID_CURR']).mean()
    df_b_filtered = df_b_filtered[['DAYS_CREDIT', 'AMT_CREDIT_SUM', 'DAYS_CREDIT_UPDATE']]
    dataframe = pd.merge(dataframe, df_b_filtered, on="SK_ID_CURR", how="left")
    del df_b
    del df_b_filtered
    return dataframe


def transform_dataframe_pos_cash_balance(dataframe):
    df_pos_cash_balance = pd.read_csv('POS_CASH_balance.csv')
    df_b_filtered = df_pos_cash_balance.groupby(['SK_ID_CURR']).mean()
    df_b_filtered = df_b_filtered[['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']]
    dataframe = pd.merge(dataframe, df_b_filtered, on="SK_ID_CURR", how="left")
    del df_pos_cash_balance
    del df_b_filtered
    return dataframe


def transform_dataframe_credit_card_balance(dataframe):
    df_ccb = pd.read_csv('credit_card_balance.csv')
    name_contract_status_map = {'Active': 1, 'Completed': 2, 'Signed': 3, 'Demand': 4, 'Sent proposal': 5, 'Refused': 6,
                                'Approved': 7}
    df_ccb.NAME_CONTRACT_STATUS = df_ccb.NAME_CONTRACT_STATUS.map(name_contract_status_map)
    df_b_filtered = df_ccb.groupby(['SK_ID_CURR']).mean()
    df_b_filtered = df_b_filtered[['MONTHS_BALANCE', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
                                   'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'CNT_INSTALMENT_MATURE_CUM', 'NAME_CONTRACT_STATUS']]
    dataframe = pd.merge(dataframe, df_b_filtered, on="SK_ID_CURR", how="left")
    del df_ccb
    del df_b_filtered
    return dataframe


def transform_dataframe_previous_application(dataframe):
    df_pa = pd.read_csv('previous_application.csv')
    df_pa["AMT_GOODS_PRICE"] = imp.fit_transform(df_pa[["AMT_GOODS_PRICE"]]).ravel()
    contract_status_map = {'Approved': 1, 'Canceled': 0, 'Refused': -1, 'Unused offer': 0.5}
    payment_type_map = {'Cash through the bank': 1, 'XNA': 2,'Non-cash from your account': 3, 'Cashless from the account of the employer': 4}
    client_status_map = {'Repeater': 1, 'New': 2, 'Refreshed': 3, 'XNA': 4}
    df_pa.NAME_CONTRACT_STATUS = df_pa.NAME_CONTRACT_STATUS.map(contract_status_map)
    df_pa.NAME_PAYMENT_TYPE = df_pa.NAME_PAYMENT_TYPE.map(payment_type_map)
    df_pa.NAME_CLIENT_TYPE = df_pa.NAME_CLIENT_TYPE.map(client_status_map)
    df_b_filtered = df_pa.groupby(['SK_ID_CURR']).mean()
    df_b_filtered = df_b_filtered[['AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 'NAME_PAYMENT_TYPE', 'NAME_CLIENT_TYPE', 'CNT_PAYMENT']]
    dataframe = pd.merge(dataframe, df_b_filtered, on="SK_ID_CURR", how="left")
    del df_pa
    del df_b_filtered
    return dataframe


def transform_dataframe_installments_payments(dataframe):
    df_ip = pd.read_csv('installments_payments.csv')
    df_b_filtered = df_ip.groupby(['SK_ID_CURR']).mean()
    df_b_filtered = df_b_filtered[['DAYS_ENTRY_PAYMENT', 'DAYS_INSTALMENT', 'AMT_PAYMENT', 'AMT_INSTALMENT']]
    dataframe = pd.merge(dataframe, df_b_filtered, on="SK_ID_CURR", how="left")
    del df_ip
    del df_b_filtered
    return dataframe