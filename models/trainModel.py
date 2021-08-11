import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from preprocessing.datacleaner import Cleaner
from sklearn.metrics import recall_score

class Predictor:
    def __init__(self, df):
        self.df = df

    def predict(self):
        self.train

    def train(self):
        X_train, X_test, y_train, y_test = self.prepare()

        results = pd.DataFrame()
        
        decisionTree = DecisionTreeClassifier.fit(X_train, y_train)
        dtAcc = recall_score(y_train, decisionTree, pos_label = 2)
        results.loc["Decision Tree"] = dtAcc

        randomForest = RandomForestClassifier.fit(X_train, y_train)
        rfAcc = recall_score(y_train, randomForest, pos_label = 2)
        results.loc["Random Forest"] = rfAcc

        gradientBoost = GradientBoostingClassifier.fit(X_train, y_train)
        gbAcc = recall_score(y_train, gradientBoost, pos_label = 2)
        results.loc["Gradient Boost"] = gbAcc

        adaBoost = AdaBoostClassifier.fit(X_train, y_train)
        abAcc = recall_score(y_train, adaBoost, pos_label = 2)
        results.loc["ADA Boost"] = abAcc

        print(results)

    def prepare(self):
        X, y = Cleaner(self.df)

        # create train and test set (random_state = 42, because it is used for official examples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # balance out the dataset with SMOTE
        balanced = SMOTE(random_state = 42)
        X_train, y_train = balanced.fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test