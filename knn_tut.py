import numpy as np
from sklearn import preprocessing, neighbors
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999 , inplace=True)
df.drop(['id'],1,inplace=True)

x=np.array(df.drop(['class'],1))
y=np.array(df['class'])

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test)
print(accuracy)

example_measure = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,3,2,3,2,1],[7,2,1,1,1,2,3,2,5],[4,2,1,1,1,2,3,2,9]])
example_measure = example_measure.reshape(len(example_measure),-1)
prediction = clf.predict(example_measure)
print(prediction)