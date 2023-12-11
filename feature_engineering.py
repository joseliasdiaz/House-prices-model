import pandas as pd
import numpy as np
from scipy.stats import boxcox

def transform(df):
  # Transfomation for missing values
  # View missing values
  null_df = (df.isnull().sum().sort_values(ascending=False) / len(df))
  null_df = null_df[null_df > 0]

  # Impute object missing values
  null_objects = df[null_df.index].select_dtypes(include='object').columns

  # Replace missing values with 'None' for pooolqc, fence, miscfeature
  for col in ['poolqc', 'fence', 'miscfeature']:
      df[col].fillna('None', inplace=True)

  # Fill missing values with mode:
  for col in null_objects:
      df[col].fillna(df[col].mode()[0], inplace=True)

  # Impute numeric missing values (median for lotfrontage, mode for the others)
  df['lotfrontage'] = df.groupby('neighborhood')['lotfrontage'].transform(lambda x: x.fillna(x.median()))
  df['garageyrblt'] = df['garageyrblt'].fillna(0)
  df['masvnrarea'] = df['masvnrarea'].fillna(0)
  df['bsmthalfbath'] = df['bsmthalfbath'].fillna(0)
  df['bsmtfullbath'] = df['bsmtfullbath'].fillna(0)
  df['bsmtfinsf1'] = df['bsmtfinsf1'].fillna(0)
  df['bsmtfinsf2'] = df['bsmtfinsf2'].fillna(0)
  df['bsmtunfsf'] = df['bsmtunfsf'].fillna(0)
  df['totalbsmtsf'] = df['totalbsmtsf'].fillna(0)
  df['garagecars'] = df['garagecars'].fillna(0)
  df['garagearea'] = df['garagearea'].fillna(0)

  # Convert MSSubClass, OverallQual, OverallCond, YrSold and MoSold to object type
  df[['mssubclass', 'overallqual', 'overallcond', 'yrsold', 'mosold']] = df[['mssubclass', 'overallqual', 'overallcond', 'yrsold', 'mosold']].astype('object')

  # Applying Box-Cox transformation to numeric columns
  # Get the numeric columns (excluding 'saleprice')
  numeric_columns = df.select_dtypes(include=[np.number]).columns.drop('saleprice')

  # Apply Box-Cox transformation to each numeric column
  for col in numeric_columns:
    df[col], _ = boxcox(df[col] + 1)  # Adding 1 to handle zero values

  # Ensure that the categorical columns exist in the dataframe
  categorical_columns = ['mssubclass', 'mszoning', 'street', 'alley', 'landcontour', 'utilities', 'lotconfig', 'neighborhood', 'condition1', 'condition2', 'bldgtype', 'housestyle', 'roofstyle', 'roofmatl', 'exterior1st', 'exterior2nd', 'masvnrtype', 'foundation', 'heating', 'centralair', 'functional', 'garagetype', 'paveddrive', 'saletype', 'salecondition']
  df[categorical_columns] = df[categorical_columns].astype('category')

  # Apply one-hot encoding to categorical columns
  df_t = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
  ordinal_columns = df_t.select_dtypes(include='object').columns
  df_t[ordinal_columns] = df_t[ordinal_columns].astype('category')

  # Apply label encoding to ordinal columns
  for col in ordinal_columns:
    df_t[col] = df_t[col].cat.codes
  return df_t