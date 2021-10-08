import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],])
clf =KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
colors = ["g","r"]

for i in range(len(X)):
    plt.scatter(X[i][0],X[i][1],s =100,c = colors[labels[i]])
plt.scatter(centroids[:,0],centroids[:,1],marker ='x')
plt.show()

