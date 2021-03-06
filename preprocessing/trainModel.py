import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score

class Predictor:
    def __init__(self, df):
        self.df = df

    def predict(self):
        self.train()

    def train(self):
        X_train, X_test, y_train, y_test = self.prepare()
        print(y_train)

        results = pd.DataFrame()
        
        decisionTree = DecisionTreeClassifier.fit(X_train, y_train)
        dtc = decisionTree.predict(X_test)
        dtAcc = recall_score(y_test, dtc, pos_label = 2)
        results["Decision Tree"] = dtAcc

        randomForest = RandomForestClassifier.fit(X_train, y_train)
        rfc = randomForest.predict(X_test)
        rfAcc = recall_score(y_test, rfc, pos_label = 2)
        results["Random Forest"] = rfAcc

        gradientBoost = GradientBoostingClassifier.fit(X_train, y_train)
        gbc = gradientBoost.predict(X_test)
        gbAcc = recall_score(y_test, gbc, pos_label = 2)
        results["Gradient Boost"] = gbAcc

        adaBoost = AdaBoostClassifier.fit(X_train, y_train)
        abc = adaBoost.predict(X_test)
        abAcc = recall_score(y_test, abc, pos_label = 2)
        results["ADA Boost"] = abAcc

        return X_train, X_test, y_train, y_test

    def prepare(self):
        # determine target and features
        y = self.df["Attrition_Flag"].to_numpy()
        X = self.df.drop("Attrition_Flag", axis = 1).to_numpy()

        # create train and test set (random_state = 42, because it is used for official examples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # balance out the dataset with SMOTE
        balanced = SMOTE(random_state = 42)
        X_train, y_train = balanced.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test