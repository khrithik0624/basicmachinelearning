import matplotlib.pyplot as plt
import numpy as np


X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],])

colors = 10*["g","r","c","b","k",]

class K_Means:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k =k
        self.tol =tol
        self.max_iter=max_iter

    def fit(self,data):
        self.centroids = {}
        self.all_centroids={}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
                self.all_centroids[i] =[]
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            prev_centroids =dict(self.centroids)
            for classification in self.classifications:
                self.all_centroids[classification].append(self.centroids[classification])
                self.centroids[classification] = np.mean(self.classifications[classification],axis=0)
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                curr_centroid = self.centroids[c]
                
                if np.sum((curr_centroid-original_centroid)/original_centroid*100.0)>self.tol:
                    optimized = False
                if optimized:
                    break
        

    def predict(self,data):
        distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker='x')

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],color=color)
for classification in clf.classifications:
    [plt.plot(clf.all_centroids[classification][i][0],clf.all_centroids[classification][i][1], marker ='*') for i in range(len(clf.all_centroids[classification])) ]
plt.show()
