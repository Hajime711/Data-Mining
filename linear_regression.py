import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

def estimatecoeff(x,y):
    n = np.size(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    sum_xy = np.sum(x*y)
    sum_xx = np.sum(x*x)
    ss_xy = sum_xy-n*mean_x*mean_y
    ss_xx = sum_xx-n*mean_x*mean_x
    b1 = ss_xy/ss_xx
    b0 = mean_y-b1*mean_x
    return (b0,b1)
def plt_regression_line(x,y,b):
    plt.scatter(x,y,color='m',marker='o',s=30)
    y_pred=b[0]+b[1]*x
    print("Accuracy:")
    print(metrics.r2_score(y, y_pred))
    plt.plot(x,y_pred,color='y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

df = pd.read_csv ('E:\Progam\Python\Salary_Data.csv')
print(df)
x = df['YearsExperience']
y = df['Salary']
b = estimatecoeff(x,y)
print("Coefficients:\nb0",b[0],'\nb1',b[1])
plt_regression_line(x,y,b)
