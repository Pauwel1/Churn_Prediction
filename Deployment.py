import pandas as pd
# from preprocessing.classification import classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from models.trainModel import decisionTree

decisionTree()


feature_descriptions = {
"Age": "Age ofthe client",
"Dependent count": "Frequency of usage",
"Relationships": "Number of past relationships",
"Months on book": "Months of delay in payment",
"Months inactive": "Months in a year the card has not been used",
"Contacts": "Number of contacts in one year",
"Credit limit": "Maximum loanable money",
"Total revolving balance": "",
"Average open to buy": "Average amount of money available",
"Total amount change Q4-Q1": "The amount loaned over year",
"Total transactions per month": "Number of times the account was used",
"Total transactions ct": "Total confidential transactions",
"Total ct changes Q4-Q1": "Amount loaned through confidential transactions over a year",
"Average utilization ratio": "Average of the card usage",
"Gender": "Male, Female",
"Education level": "College, Doctorate, Graduate, High School, Post-Graduate, Uneducated, Unknown",
"Marital status": "Divorced, Married, Single, Unknown",
"Income category": "$120K +, $40K - $60K, $60K - $80K, $80K - $120K, Less than $40K, Unknown",
"Card category": "Blue, Gold, Platinum, Silver"
}
    
explainer = ClassifierExplainer(model, X_train, y_pred)

db = ExplainerDashboard(explainer, title = "Churn Explainer", shap_interaction = False)

db.run(port = 8000)