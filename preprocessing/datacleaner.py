import pandas as pd
import numpy as np

# import dataframe
crunch = pd.read_csv("assets/BankChurners.csv")

# data cleaning
# drop last two biassed columns
crunch = crunch[crunch.columns[:-2]]

# change dtypes
crunch["Gender"] = crunch["Gender"].astype(str)

print("dtypes: \n", crunch.dtypes)
print("__________")

print("__________")
print(crunch["Gender"].value_counts().sum())