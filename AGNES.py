import numpy as np
import matplotlib.pyplot as mtp
import pandas as ab
Datasets = ab.read_csv('C:/Users/ADMIN/Desktop/Positive/ManagerDetails.csv')
print(Datasets)
A = Datasets.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(A, method="ward"))
mtp.title("The Dendogram Graph :")
mtp.ylabel("This is an Euclidean Distance Part")
mtp.xlabel("These are the Managers")
mtp.show()
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward')
y_predict = hc.fit_predict(A)
mtp.scatter(A[y_predict == 0, 0], A[y_predict == 0, 1], s = 70, c='yellow',label =
'The First Cluster')
mtp.scatter(A[y_predict == 1, 0], A[y_predict == 1, 1], s = 70, c='green',label =
'The Second Cluster')
mtp.scatter(A[y_predict == 2, 0], A[y_predict == 2, 1], s = 70, c='Blue',label =
'The Third Cluster')
mtp.title("The Three Clusters of Managers" )
mtp.xlabel("Annual Income of Manager(k$)")
mtp.ylabel("Manager Spending Score From (1-100)")
mtp.legend()
mtp.show()
