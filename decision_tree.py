from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("E:\Progam\Python\zoo.csv")
df.head()
df['animal'],_ = pd.factorize(df['animal'])
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("Accuracy Score (Decision Tree):" , accuracy_score(y_test,y_pred))

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

print('Accuracy score (Naive Bayes):' ,(accuracy_score(y_test, y_pred)))
