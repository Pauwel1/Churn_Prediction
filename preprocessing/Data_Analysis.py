import pandas as pd
import numpy as np

crunch = pd.read_csv("assets/BankChurners.csv")

print("dtypes: \n", crunch.dtypes)
print("__________")
print(crunch.isnull().sum())
print("__________")
print(crunch["Gender"].value_counts().sum())