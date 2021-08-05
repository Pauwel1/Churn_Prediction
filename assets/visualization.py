import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

crunch = pd.read_csv("assets/BankChurners.csv")

# data cleaning
# drop columns that are not necessary or don't add value
crunch = crunch[crunch.columns[:-2]]
crunch = crunch.drop("CLIENTNUM", axis = 1)

corr = crunch.corr()
sns.heatmap(corr)
plt.tight_layout()
plt.savefig("assets/heatmap.png")
plt.clf()

sns.scatterplot(x = crunch["Avg_Open_To_Buy"], y = crunch["Months_on_book"], data = crunch, hue = crunch["Attrition_Flag"])
plt.savefig("assets/scatter1.png")
plt.clf()
