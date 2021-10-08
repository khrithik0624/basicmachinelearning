import matplotlib.pyplot as plt
import numpy as np


X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2],[9,3],])

colors = 10*["g","r","c","b","k",]
plt.scatter(X[:,0],X[:,1],s=100)
plt.show()

class MeanShift:
	def __init__(self,radius=4):
		self.radius = radius

	def fit(self, data):
		centroids = {}
		for i in range(len(data)):
			centroids[i]= data[i]

		while True:
			new_centroids = []
			for i in centroids:
				in_radius = []
				centroid =centroids[i]
				for feature_set in data:
					if np.linalg.norm(feature_set - centroid) < self.radius:
						in_radius.append(feature_set) 

				new_centroid =np.average(in_radius,axis=0)
				new_centroids.append((tuple(new_centroid)))
			uniques = sorted(list(set(new_centroids)))
			prev_centroids = dict(centroids)
			centroids  = {}
			for i in range(len(uniques)):
				centroids[i] =np.array(uniques[i])

			optimized = True 
			for i in centroids:
			 	if not np.array_equal(prev_centroids[i], centroids[i]):
			 		optimized =False
			 		break
			if optimized:
				break

		self.centroids = centroids

	def predict(self,data):
		for centroid in self.centroids:
			if np.linalg.norm(data - centroids[centroid]) < self.radius:
				return centroid


clf = MeanShift()
clf.fit(X)
centroids = clf.centroids
[plt.scatter(X[i][0],X[i][1],s=100,color= colors[clf.predict(X[i])] ) for i in range(len(X))]
[plt.scatter(centroids[i][0],centroids[i][1],marker ='x',s=100,color='k') for i in range(len(centroids))]
plt.show()

