import pandas as pd

df = pd.read_csv("chicago/Chicago_Crimes_2012_to_2017.csv")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# print(df.columns)

df.drop(["ID", "Case Number","Description","FBI Code","Updated On", "Year","Date","Block" ,'Location Description', 'X Coordinate', 'Y Coordinate',  'Community Area', 'Location','IUCR', "Longitude","Y Coordinate","X Coordinate","Latitude"], inplace = True, axis=1)
# print(df.isnull().sum())
nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(subset=["District", "Ward"], inplace=True)
df.drop_duplicates()


le = LabelEncoder()
le.fit(df["Arrest"])
df["Arrest"] = le.transform(df["Arrest"])

df["Primary Type"] = le.fit_transform(df["Primary Type"])


# print(df["Arrest"])
x = df.drop(["Arrest"], axis=1)
y = df["Arrest"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

rfc = RandomForestClassifier(random_state=1)
logr = LogisticRegression(random_state=0)
gbc = GradientBoostingClassifier(n_estimators=10)
dtc = DecisionTreeClassifier(random_state=0)
svm = svm.SVC()
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)


rfc.fit(x_train, y_train)
y_rfc =rfc.predict(x_test)

logr.fit(x_train, y_train)
y_logr = logr.predict(x_test)

gbc.fit(x_train, y_train)
y_gbc = gbc.predict(x_test)

dtc.fit(x_train, y_train)
y_dtc = dtc.predict(x_test)

svm.fit(x_train, y_train)
y_svm = svm.predict(x_test)

nn.fit(x_train, y_train)
y_nn = nn.predict(x_test)



print("Random Forest:", accuracy_score(y_test, y_rfc))
print("Logistic Regression:", accuracy_score(y_test, y_logr))
print("Gradient Boosting:", accuracy_score(y_test, y_gbc))
print("Decision Tree:", accuracy_score(y_test, y_dtc))
print("Support Vector Machine:", accuracy_score(y_test, y_svm))
print("Artificial Neural Network:", accuracy_score(y_test, y_nn))
