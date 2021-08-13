import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from imblearn.over_sampling import SMOTE

churn = pd.read_csv("assets/BankChurners.csv")

# data cleaning
# drop columns that are not necessary or don't add value
churn = churn[churn.columns[:-2]]
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

churn.loc[churn['Income_Category'] == 'Less than $40K', "Income_Category"] = 30
churn.loc[churn['Income_Category'] == '$40K - $60K', "Income_Category"] = 50
churn.loc[churn['Income_Category'] == '$60K - $80K', "Income_Category"] = 70
churn.loc[churn['Income_Category'] == '$80K - $120K', "Income_Category"] = 100
churn.loc[churn['Income_Category'] == '$120K +', "Income_Category"] = 200
churn.loc[churn['Income_Category'] == 'Unknown', "Income_Category"] = churn['Income_Category'].mode()[0]
churn["Income_Category"] = churn["Income_Category"].astype(int)

churn.loc[churn['Card_Category'] == 'Blue', "Card_Category"] = 222
churn.loc[churn['Card_Category'] == 'Silver', "Card_Category"] = 333
churn.loc[churn['Card_Category'] == 'Gold', "Card_Category"] = 444
churn.loc[churn['Card_Category'] == 'Platinum', "Card_Category"] =  555
churn["Card_Category"] = churn["Card_Category"].astype(int)

churn.loc[churn["Marital_Status"] == "Married", "Marital_Status"] = 120
churn.loc[churn["Marital_Status"] == "Single", "Marital_Status"] = 140
churn.loc[churn["Marital_Status"] == "Divorced", "Marital_Status"] = 160
churn.loc[churn["Marital_Status"] == "Unknown", "Marital_Status"] = churn["Marital_Status"].mode()[0]
churn["Marital_Status"] = churn["Marital_Status"].astype(int)

churn.loc[churn["Gender"] == "M", "Gender"] = 1
churn.loc[churn["Gender"] == "F", "Gender"] = 2
churn["Gender"] = churn["Gender"].astype(int)

# determine target and features
y = churn["Attrition_Flag"].to_numpy()
X = churn.drop("Attrition_Flag", axis = 1)

print(np.unique(y))

# create train and test set (random_state = 42, because it is used for official examples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# balance out the dataset with SMOTE
balanced = SMOTE(random_state = 42)
X_train, y_train = balanced.fit_resample(X_train, y_train)

# print(X_train.columns)

model = RandomForestClassifier(max_depth=4).fit(X_train, y_train)

# feature_descriptions = {"Avg_Utilization_Ratio": "",
# "Card_Category_0": "",
# "Card_Category_1": "",
# "Contacts_Count_12_mon": "",
# "Credit_Limit": "",
# "Customer_Age": "",
# "Dependent_count": "",
# "Education_Level_0": "",
# "Education_Level_1": "",
# "Gender_0": "",
# "Gender_1": "",
# "Income_Category_0": "",
# "Income_Category_1": "",
# "Marital_Status_0":
# Marital_Status_1	
# Months_Inactive_12_mon	
# Months_on_book	
# Total_Amt_Chng_Q4_Q1	
# Total_Ct_Chng_Q4_Q1	
# Total_Relationship_Count	
# Total_Revolving_Bal	
# Total_Trans_Amt	
# Total_Trans_Ct
# }

explainer = ClassifierExplainer(model, X_test, y_test,
                            #    descriptions = feature_descriptions,
                               labels=['Attrited_Customer', 'Existing_Customer'])

ExplainerDashboard(explainer).run(host='0.0.0.0', port=9050, use_waitress=True)