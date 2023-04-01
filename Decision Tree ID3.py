import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
itemsets = pd.read_csv("C:/Users/ADMIN/Desktop/Positive/Road Transport Dataset.csv")
print(itemsets)
y={'no': 0,'yes':1}
itemsets['Accident_Risk']=itemsets['Accident_Risk'].map(y)
print(itemsets)
Train =['Road_ID','Length','Number_Of_Bends','Traffic_Volume']
x=itemsets[Train]
y=itemsets['Accident_Risk']
print(x)
print(y)
DTC= DecisionTreeClassifier()
DTC=DTC.fit(x,y)
z=DTC.predict([[1015,12000,12,8500]])
print(z)
z=DTC.predict([[1014,16500,16,7500]])
print(z)
tree.plot_tree(DTC)
