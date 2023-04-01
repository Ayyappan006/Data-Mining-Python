import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
datasets = pd.read_csv("C:/Users/ADMIN/Desktop/Positive/HealthDetails.csv")
print(datasets)
from sklearn.cluster import KMeans
r = {'weak' : 0,'normal' : 1,'healthy' : 2}
datasets['result'] = datasets['result'].map(r)
within_Cluster_Sum_of_Squares = []
for proposed_number_of_clusters in range(1,11):
 kmeans = KMeans(n_clusters=proposed_number_of_clusters,
random_state= 42)
 kmeans.fit(datasets)
 within_Cluster_Sum_of_Squares.append(kmeans.inertia_)
fig, axis = plt.subplots(figsize =(11,8))
sns.lineplot(x=range(1,11),y=within_Cluster_Sum_of_Squares, ax= axis)
plt.title("ELBOW METHOD")
plt.ylabel("Within The Cluster Sum of Squares Error")
plt.xlabel("The Number of Clusters")
plt.show()
kmeans = KMeans(n_clusters=3)
kmeans_predictions = kmeans.fit_predict(datasets)
datasets = datasets.values
print("So,These are the labels Which is predicting :")
(kmeans.labels_)
fig. axis = plt.subplots(figsize =(11,8))
plt.scatter(datasets[kmeans_predictions
==0,0],datasets[kmeans_predictions==0, 1],c="yellow", s=100, label= "The First
Cluster")
plt.scatter(datasets[kmeans_predictions
==1,0],datasets[kmeans_predictions==1, 1],c="blue", s=100, label= "The
Second Cluster")
plt.scatter(datasets[kmeans_predictions
==2,0],datasets[kmeans_predictions==2, 1],c="green", s=100, label= "The Third
Cluster")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100,
label ="The Centroids", c="red")
plt.legend()
print("These are the BMI'S in the First Clusters :-")
print(datasets[kmeans_predictions == 0,2])
print("These are the BMI'S in the Second Clusters :-")
print(datasets[kmeans_predictions == 1, 2])
print("These are the BMI'S in the Third Clusters :-")
print(datasets[kmeans_predictions == 2, 2])
