import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

def k_nearest_neighbors(data, predict, k=3):
    if len(data)>=k:
        warnings.warn('k is set to value lower than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
            #euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2) )
            euclidean_distance = np.linalg.norm( np.array(features)- np.array(predict) )
            distances.append([euclidean_distance,group])
    
    votes = [i[1] for i in sorted(distances) [:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

dataset = {'k':[[1,2,1],[2,3,8],[3,1,4]], 'r':[[6,5,1],[7,7,8],[8,6,4]]}
new_features = [5,7,6]
result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)
#for i in dataset:
#    for ii in dataset[i]:
#        plt.scatter(ii[0],ii[1],s=100,color=i)
ax = plt.axes(projection ="3d")
[[ax.scatter3D(ii[0],ii[1],ii[2],s=100,color=i) for ii in dataset[i]] for i in dataset ]
ax.scatter3D(new_features[0],new_features[1],new_features[2],s=100,color= result)
plt.show()

