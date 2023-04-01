import pandas as pd
import numpy as np
datasets=pd.read_csv("C:/Users/ADMIN/Desktop/Positive/Student.csv")
print(datasets)
datasets.tail()
x= datasets.iloc[:, 0:3].values
y= datasets.iloc[:, 3].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
x_train = SC.fit_transform(x_train)
x_test = SC.transform(x_test)
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor()
Regressor.fit(x_train, y_train)
y_predict = Regressor.predict(x_test)
Regressor.predict([['124','10','399']])
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor()
Regressor.fit(x_train, y_train)
y_predict = Regressor.predict(x_test)
Regressor.predict([['106','10','345']])
from sklearn import metrics
print('The Mean Absolute Error:',metrics.mean_absolute_error(y_test ,y_predict))
print('The mean Squared Error:',metrics.mean_squared_error(y_test,y_predict))
print('The Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
