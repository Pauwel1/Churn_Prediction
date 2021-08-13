import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from datacleaner import Cleaner

def classifier(classifier, mdl):
  churn = Cleaner()
  churn = churn.dataCleaner()

  # determine target and features
  y = churn["Attrition_Flag"].to_numpy()
  X = churn.drop("Attrition_Flag", axis = 1)

  # create train and test set (random_state = 42, because it is used for official examples)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

  # balance out the dataset with SMOTE
  balanced = SMOTE(random_state = 42)
  X_bal, y_bal = balanced.fit_resample(X_train, y_train)

  # Try out different classifiers
  model = classifier.fit(X_bal, y_bal)

  y_hat = model.predict(X_train)
  acc = recall_score(y_train, y_hat, pos_label = 0)
  results.loc[mdl, 'Train'] = acc

  y_hat2 = model.predict(X_test)
  acc = recall_score(y_test, y_hat2, pos_label = 1)
  results.loc[mdl, 'Test'] = acc

# Storing Results
results = pd.DataFrame()

# Models
stage = 'Classification Model'
classifier(DecisionTreeClassifier(), 'Decision Tree')
classifier(RandomForestClassifier(), 'Random Forest')
classifier(GradientBoostingClassifier(), 'Gradient Boost')
classifier(AdaBoostClassifier(), 'AdaBoost')
print(results)