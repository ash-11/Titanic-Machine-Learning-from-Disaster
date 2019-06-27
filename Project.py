import pandas as pd
import numpy as np

train = pd.read_csv('./train.csv', low_memory= False)
test = pd.read_csv('./test.csv', low_memory = False)

train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)

train.drop('PassengerId', axis = 1, inplace = True)
train.drop('Cabin', axis = 1, inplace = True)
train.drop('Ticket', axis = 1, inplace = True)
test.drop('PassengerId', axis = 1, inplace = True)
test.drop('Cabin', axis = 1, inplace = True)
test.drop('Ticket', axis = 1, inplace = True)

train['Age'].fillna(train['Age'].mean(), inplace = True)
test['Age'].fillna(test['Age'].mean(), inplace = True)
test['Fare'].fillna(test['Fare'].mean(), inplace = True)

train = train.dropna()

Pclass = pd.get_dummies(train['Pclass'], drop_first = True)
Pclass1 = pd.get_dummies(test['Pclass'], drop_first = True)
Sex = pd.get_dummies(train['Sex'], drop_first = True)
Sex1 = pd.get_dummies(test['Sex'], drop_first = True)
Embarked = pd.get_dummies(train['Embarked'], drop_first = True)
Embarked1 = pd.get_dummies(test['Embarked'], drop_first = True)

train = pd.concat([train, Pclass, Sex, Embarked], axis = 1)
test = pd.concat([test, Pclass1, Sex1, Embarked1], axis = 1)

train.drop(['Pclass', 'Sex', 'Embarked'], axis = 1, inplace = True)
test.drop(['Pclass', 'Sex', 'Embarked'], axis = 1, inplace = True)

y = train['Survived']
x = train.drop(['Survived'], axis = 1)

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(x, y)
pred = clf.predict(test)

sam = pd.read_csv('./gender_submission.csv', low_memory=False)
sam['Survived'] = pred
sam.to_csv('pred.csv', index = False)