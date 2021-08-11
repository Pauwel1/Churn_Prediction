import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Cleaner:
    def __init__(self):
        self.df = pd.read_csv("assets/BankChurners.csv")

    def dataCleaner(self):
        # data cleaning
        # drop columns that are not necessary or don't add value
        self.df = self.df[self.df.columns[:-2]]
        self.df = self.df.drop("CLIENTNUM", axis = 1)

        # check NaN values
        print(self.df[self.df.isnull()].count())

        # change target values into numericals
        self.df[self.df["Attrition_Flag"] == 'Existing Customer'] = 1
        self.df[self.df["Attrition_Flag"] == "Attrited Customer"] = 2
        self.df["Attrition_Flag"] = self.df["Attrition_Flag"].astype(int)

        # create dummies of categorical features
        # (all are object values -> select_dtypes)
        cat_columns = self.df.select_dtypes(['object'])

        for item in cat_columns:
            dummies = pd.get_dummies(self.df[item], columns = cat_columns.columns, prefix = item)
            self.df = pd.concat([self.df, dummies], axis = 1)
            del self.df[item]
        
        return self.df

    # def visualize(self):
    #     features = ['Customer_Age', 'Months_on_book', 'Total_Relationship_Count', 'Dependent_count',
    #                 'Avg_Utilization_Ratio', 'Months_Inactive_12_mon', 'Total_Trans_Amt', 'Credit_Limit']
    #     for feature in features:
    #         fig, axes = plt.subplots(2, 1)
    #         sns.boxplot(x=self.df[feature], showmeans=True, ax=axes[0]).set_title('Box Plot')
    #         sns.histplot(x=self.df[feature], ax=axes[1]).set_title('Histogram')
    #         plt.tight_layout()
    #         fig.suptitle('Analyzing ' + feature)
    #         stl.pyplot()