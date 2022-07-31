from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

df = pd.read_csv("Python/final_test.csv")
print("Sample of DataSet:")
print(df.head())
plt.hist(df["size"],20)
df = df.fillna(0)
columns = ["weight","age","height"]
x = df[columns].values
y = df['size']
#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)
model = KNeighborsClassifier(n_neighbors=3)
model = model.fit(x_train,y_train)
y_pre = model.predict(x_test)
print("Prediction : ",y_pre)
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pre))
print("Accuracy Score:",accuracy_score(y_test,y_pre))

plt.show()
