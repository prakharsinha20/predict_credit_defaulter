import pandas as pd
#import xgboost as xgb
from dataframe_transformation_methods import *


def data_cleaning(dataframe):
    dataframe = map_and_impute(dataframe)
    dataframe = transform_dataframe_bureau(dataframe)
    dataframe = transform_dataframe_pos_cash_balance(dataframe)
    dataframe = transform_dataframe_credit_card_balance(dataframe)
    dataframe = transform_dataframe_previous_application(dataframe)
    dataframe = transform_dataframe_installments_payments(dataframe)
    return dataframe


attributes = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
              'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
              'NAME_FAMILY_STATUS', 'DAYS_EMPLOYED']
df = pd.read_csv('abc.csv')
df_X = df[attributes]
df_X = data_cleaning(df_X)
df_X = df_X.fillna(df_X.mean())
df_Y = df['TARGET']
gbm = xgb.XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05).fit(df_X, df_Y)
df_test = pd.read_csv('application_test.csv')
df_test_X = df_test[attributes]
df_test_X = data_cleaning(df_test_X)
predictions = gbm.predict(df_test_X)
submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': predictions})
submission.to_csv('submission.csv', index=False)
print('Predictions Done!!!!!!!!!!!!!!!')