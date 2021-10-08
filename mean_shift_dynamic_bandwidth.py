import matplotlib.pyplot as plt
import numpy as np


X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2],[9,3],])

colors = 10*["g","r","c","b","k",]
plt.scatter(X[:,0],X[:,1],s=100)
plt.show()

class MeanShift:
	def __init__(self,radius=None,radius_step=100):
		self.radius = radius
		self.radius_step =radius_step

	def fit(self, data):
		
		if self.radius==None:
			all_data_centroid = np.average(data,axis=0)
			all_data_norm = np.linalg.norm(all_data_centroid)
			self.radius = all_data_norm/self.radius_step

		centroids = {}
		for i in range(len(data)):
			centroids[i]= data[i]

		weights = [i for i in range(self.radius_step)][::-1]

		while True:
			new_centroids = []
			for i in centroids:
				in_radius = []
				centroid =centroids[i]

				for feature_set in data:
					distance = np.linalg.norm(feature_set-centroid)
					if distance == 0:
						distance = 0.0000001
					weight_index = int(distance/self.radius)
					if weight_index > self.radius_step -1:
						weight_index = self.radius_step-1
					to_add= (weights[weight_index]**2)*[feature_set]
					in_radius += to_add


				new_centroid =np.average(in_radius,axis=0)
				new_centroids.append((tuple(new_centroid)))
			uniques = sorted(list(set(new_centroids)))

			to_pop = []
			for i in uniques:
				for ii in uniques:
					if i==ii:
						pass
					elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
						to_pop.append(ii)
						break

			for i in to_pop:
				try:
					unique.remove(i)
				except:
					pass


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

		self.classifications = {}
		for i in range(len(self.centroids)):
			self.classifications[i] = []

		for feature_set in data:
			distances = [np.linalg.norm(feature_set-self.centroids[centroid]) for centroid in self.centroids]
			classification = distances.index(min(distances))
			self.classifications[classification].append(feature_set)

	def predict(self,data):
		distances = [np.linalg.norm(feature_set-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

clf = MeanShift()
clf.fit(X)
centroids = clf.centroids
[plt.scatter(centroids[i][0],centroids[i][1],marker ='x',s=100,color='k') for i in range(len(centroids))]
for classification in clf.classifications:
	color = colors[classification]
	[plt.scatter(feature[0],feature[1],color=color ) for feature in clf.classifications[classification]]
plt.show()