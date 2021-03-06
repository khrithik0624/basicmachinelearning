import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

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
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace =True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0
record = []

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        record.append([vote,group])
        if vote == group:
            correct += 1
        total += 1

print(correct/total*100)
print(record)