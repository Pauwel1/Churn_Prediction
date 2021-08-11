import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Processor:
    def __init__(self):
        self.df = pd.read_csv("assets/BanckChurners.csv")

    def dataCleaner(self):
        # data cleaning
        # drop columns that are not necessary or don't add value
        df = self.df[self.df.columns[:-2]]
        df = self.df.drop("CLIENTNUM", axis = 1)

        # # check NaN values
        # print(df[df.isnull()].count())

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

    def visualize(self):
        features = ['Customer_Age', 'Months_on_book', 'Total_Relationship_Count', 'Dependent_count',
                    'Avg_Utilization_Ratio', 'Months_Inactive_12_mon', 'Total_Trans_Amt', 'Credit_Limit']
        for feature in features:
            fig, axes = plt.subplots(2, 1)
            sns.boxplot(x=self.df[feature], showmeans=True, ax=axes[0]).set_title('Box Plot')
            sns.histplot(x=self.df[feature], ax=axes[1]).set_title('Histogram')
            plt.tight_layout()
            # fig.suptitle('Analyzing ' + feature)
            # stl.pyplot()