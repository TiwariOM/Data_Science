import pandas as pd
df = pd.read_csv('train.csv')
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df['Age']=(df['Age'].str.strip('+'))
df['Stay_In_Current_City_Years']=(df['Stay_In_Current_City_Years'].str.strip('+').astype('float'))

df['Product_Category_2'].fillna(df['Product_Category_2'].mean(),inplace=True)
df['Product_Category_3'].fillna(df['Product_Category_3'].mean(),inplace=True)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Age'] = le.fit_transform(df['Age'])
df['City_Category'] = le.fit_transform(df['City_Category'])

x = df.drop(['Purchase','Product_ID','User_ID'], axis=1)
x = x.astype('int')
y = df['Purchase']


svc = SVR()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3 )

svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

print(mean_squared_error(y_test,y_pred)