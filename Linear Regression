import pandas as ab
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
datasets = ab.read_csv("C:/Users/ADMIN/Desktop/Positive/COVID-19 Cases.csv")
month = datasets.month
cases = datasets.cases
plt.xlabel('month')
plt.ylabel('cases')
plt.scatter(month,cases)
plt.show()
r1=linear_model.LinearRegression()
r1.fit(datasets[['month']],datasets.cases)
r1.predict([[18]])
# Co-efficient value
print(r1.coef_)
# Intercept value
print(r1.intercept_)
slope = r1.coef_
intercept = r1.intercept_
print(slope)
print(intercept)
month = datasets.month
cases = datasets.cases
plt.scatter(month,cases)
plt.plot(month,slope*month+intercept,color='red')
plt.show()
