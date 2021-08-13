import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Cleaner:
    def __init__(self):
        self.df = pd.read_csv("assets/BankChurners.csv")

    def dataCleaner(self):
        # data cleaning
        # drop columns that are not necessary or don't add value
        churn = self.df[self.df.columns[:-2]]
        churn = churn.drop("CLIENTNUM", axis = 1)

        # check NaN values
        print(churn[churn.isnull()].count())

        # drop columns that are too correlated
        # Create correlation matrix
        corr_matrix = churn.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
        # Drop features 
        churn.drop(to_drop, axis=1, inplace=True)

        # change target values into numericals
        churn.loc[churn["Attrition_Flag"] == 'Existing Customer', "Attrition_Flag"] = 1
        churn.loc[churn["Attrition_Flag"] == "Attrited Customer", "Attrition_Flag"] = 0
        churn["Attrition_Flag"] = churn["Attrition_Flag"].astype(int)
        print(churn["Attrition_Flag"].unique())

        # change string values to numeric
        churn.loc[churn['Education_Level'] == 'College', "Education_Level"] = 2
        churn.loc[churn['Education_Level'] == 'Doctorate', "Education_Level"] = 5
        churn.loc[churn['Education_Level'] == 'Graduate', "Education_Level"] = 3
        churn.loc[churn['Education_Level'] == 'High School', "Education_Level"] = 1
        churn.loc[churn['Education_Level'] == 'Post-Graduate', "Education_Level"] = 4
        churn.loc[churn['Education_Level'] == 'Uneducated', "Education_Level"] = 0
        churn.loc[churn['Education_Level'] == 'Unknown', "Education_Level"] = churn['Education_Level'].mode()[0]
        churn["Education_Level"] = churn["Education_Level"].astype(int)

        churn.loc[churn['Income_Category'] == 'Less than $40K', "Income_Category"] = 1
        churn.loc[churn['Income_Category'] == '$40K - $60K', "Income_Category"] = 2
        churn.loc[churn['Income_Category'] == '$60K - $80K', "Income_Category"] = 3
        churn.loc[churn['Income_Category'] == '$80K - $120K', "Income_Category"] = 4
        churn.loc[churn['Income_Category'] == '$120K +', "Income_Category"] = 5
        churn.loc[churn['Income_Category'] == 'Unknown', "Income_Category"] = churn['Income_Category'].mode()[0]
        churn["Income_Category"] = churn["Income_Category"].astype(int)

        churn.loc[churn['Card_Category'] == 'Blue', "Card_Category"] = 1
        churn.loc[churn['Card_Category'] == 'Silver', "Card_Category"] = 2
        churn.loc[churn['Card_Category'] == 'Gold', "Card_Category"] = 3
        churn.loc[churn['Card_Category'] == 'Platinum', "Card_Category"] =  4
        churn["Card_Category"] = churn["Card_Category"].astype(int)

        churn.loc[churn["Marital_Status"] == "Married", "Marital_Status"] = 1
        churn.loc[churn["Marital_Status"] == "Single", "Marital_Status"] = 2
        churn.loc[churn["Marital_Status"] == "Divorced", "Marital_Status"] = 3
        churn.loc[churn["Marital_Status"] == "Unknown", "Marital_Status"] = churn["Marital_Status"].mode()[0]
        churn["Marital_Status"] = churn["Marital_Status"].astype(int)

        churn.loc[churn["Gender"] == "M", "Gender"] = 1
        churn.loc[churn["Gender"] == "F", "Gender"] = 2
        churn["Gender"] = churn["Gender"].astype(int)

        return churn

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