import pandas as pd
import numpy as np
from pandas.core.series import Series

# import dataframe
crunch = pd.read_csv("assets/BankChurners.csv")

# data cleaning
# drop columns that are not necessary
crunch = crunch[crunch.columns[:-2]]
crunch = crunch.drop("CLIENTNUM", axis = 1)

# determine target and features
y = crunch["Attrition_Flag"].to_numpy()
X = crunch.drop("Attrition_Flag", axis = 1)

# create dummies
cat_columns = X.select_dtypes(include = ['object'])

for item in cat_columns:
    dummies = pd.get_dummies(X[item], columns = cat_columns.columns)
    X = pd.concat([X, dummies], axis = 1)
    del X[item]

print(X.columns)