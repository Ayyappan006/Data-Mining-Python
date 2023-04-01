import pandas as ab
import numpy as np
import matplotlib.pyplot as p1
from sklearn import preprocessing
datasets = ab.read_csv("C:/Users/ADMIN/Desktop/Positive/Diet maintenance.csv")
LE = preprocessing.LabelEncoder()
x=datasets.iloc[:,[1,2,3]].values
y=datasets.iloc[:,[-1]].values.flatten()
print(datasets)
x[:,0] = LE.fit_transform(x[:,0])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x, y,test_size=0.20,random_state=0)
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
x_train = SC.fit_transform(x_train)
x_test = SC.transform(x_test)
from sklearn.naive_bayes import GaussianNB
Classifier = GaussianNB()
Classifier.fit(x_train, y_train)
y_predict = Classifier.predict(x_test)
print(y_predict)
from sklearn.metrics import confusion_matrix,accuracy_score
CM =confusion_matrix(y_test, y_predict)
AC =accuracy_score(y_test,y_predict)
# 0-no,1-yes
print(AC) print(CM)
