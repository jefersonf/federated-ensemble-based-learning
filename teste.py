from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 
from models.ensemble_model import EnsembleClassifier

np.random.seed(1)

# prepare data
dataset = pd.read_csv('./dataset/diabetes.csv')

X = dataset.drop('Outcome',axis=1)
y = dataset['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

decision_tree = DecisionTreeClassifier()
reg = linear_model.LogisticRegression()

decision_tree.fit(X_train,y_train)
reg.fit(X_train,y_train)


print("Decision tree metrics")
a = X_test.loc[(X_test['SkinThickness'] == 36)]

y_pred1 = decision_tree.predict_proba(a)
print("###################")
print(y_pred1)

# print(confusion_matrix(y_test, y_pred1))
# print(classification_report(y_test, y_pred1))

print("Linear regression metrics")
y_pred2 = reg.predict_proba(a)
print(y_pred2)
# print(confusion_matrix(y_test, y_pred2))
# print(classification_report(y_test, y_pred2))
print("###################")
print("Combined metrics")

ensemble_model = EnsembleClassifier([decision_tree, reg],weights = [1,2])
y_pred3 = ensemble_model._predict_proba(a)
print(y_pred3)
# print(confusion_matrix(y_test, y_pred3))
# print(classification_report(y_test, y_pred3))