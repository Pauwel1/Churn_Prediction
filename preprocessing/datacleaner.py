import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataCleaner(df):
    # data cleaning
    # drop columns that are not necessary or don't add value
    df = df[df.columns[:-2]]
    df = df.drop("CLIENTNUM", axis = 1)

    # check NaN values
    print(df[df.isnull()].count())

    # determine target and features
    y = df["Attrition_Flag"].to_numpy()
    X = df.drop("Attrition_Flag", axis = 1)

    # change target values into numericals
    y[y == 'Existing Customer'] = 1
    y[y == "Attrited Customer"] = 2
    y = y.astype(int)

    # create dummies of categorical features
    # (all are object values -> select_dtypes)
    cat_columns = X.select_dtypes(['object'])

    for item in cat_columns:
        dummies = pd.get_dummies(X[item], columns = cat_columns.columns, prefix = item)
        X = pd.concat([X, dummies], axis = 1)
        del X[item]

    return X, y

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
