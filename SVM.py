import pandas as ab
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
datasets = ab.read_csv("C:/Users/ADMIN/Desktop/Positive/Student 9 Pointer.csv")
print(datasets)
print(datasets.iloc[3])
print(datasets.iloc[:1:20])
print(datasets.iloc[1:20])
training_items,test_items=train_test_split(datasets,test_size=0.2)
x_train= training_items.iloc[:,0:2].values
x_test=test_items.iloc[:,0:2].values
print(x_test)
print(x_train)
y_train=training_items.iloc[:,2].values
y_test=test_items.iloc[:,2].values
print(y_test)
print(y_train)
S=SVC(kernel='linear')
S.fit(x_train,y_train)
S.predict([[96,90]])
