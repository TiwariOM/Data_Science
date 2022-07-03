import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.isnull().sum()

X = train[['Dept','IsHoliday']].loc[ (train['Store']==1) ]
Y = train[['Weekly_Sales']].loc[ (train['Store']==1) ]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

le=LabelEncoder()
le.fit(Y_train)

LinReg = LinearRegression()
LinReg.fit(X_train, Y_train)

train6 = LinReg.fit(X_train,Y_train)
Y_pred6=LinReg.predict(X_test)
print(LinReg.score(X_test, Y_test))

print(f'Intercept = {LinReg.intercept_} \nCoefficient = {len(LinReg.coef_)}')

print(f'Mean Squared Error = {np.mean((LinReg.predict(X_test)-Y_test)**2)}'
      + f'\nVariance Score = {LinReg.score(X_test,Y_test)}')
