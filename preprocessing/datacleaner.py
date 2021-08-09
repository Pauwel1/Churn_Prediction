import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataCleaner(df):
    # data cleaning
    # drop columns that are not necessary or don't add value
    df = df[df.columns[:-2]]
    df = df.drop("CLIENTNUM", axis = 1)

    # check NaN values
    print(df[df.isnull()].count())

    return df

def visualizer(df):
    # plot heatmap
    corr = df.corr()
    sns.heatmap(corr)
    plt.tight_layout()
    plt.savefig("assets/heatmap.png")
    plt.clf()

    #plot scatterplot
    sns.scatterplot(x = df["Avg_Open_To_Buy"], y = df["Months_on_book"], data = df, hue = df["Attrition_Flag"])
    plt.savefig("assets/scatter1.png")
    plt.clf()
