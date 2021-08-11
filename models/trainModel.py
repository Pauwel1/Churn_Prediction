import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from preprocessing.datacleaner import dataCleaner
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

class Predictor:
    def __init__(self, df):
        self.df = df

    def predict(self):
        self.train

    def train(self):
       X_train, X_test, y_train, y_test = self.prepare()

       decisionTree = DecisionTreeClassifier.fit(X_train, y_train)

       randomForest = RandomForestClassifier.fit(X_train, y_train)

       gradientBoost = GradientBoostingClassifier.fit(X_train, y_train)

       adaBoost = AdaBoostClassifier.fit(X_train, y_train)

    def prepare(self):
        X, y = dataCleaner(self.df)

        # create train and test set (random_state = 42, because it is used for official examples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # balance out the dataset with SMOTE
        balanced = SMOTE(random_state = 42)
        X_train, y_train = balanced.fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test