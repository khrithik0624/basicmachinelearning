import pandas as pd
import quandl , math , datetime
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,  svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
#print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume',]]
#print(df.head())

df['HL_pct'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0
df['pct_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_pct','pct_change','Adj. Volume']]
#print(df.head())

forecast_col= 'Adj. Close'
df.fillna(-999999,inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
#print(df.head())

#features
x = np.array(df.drop(['label'],1))
#labels

x = preprocessing.scale(x)

print(len(x))
x_lately = x[-forecast_out:]
print(len(x_lately))
df.dropna(inplace=True)
y = np.array(df['label'])
print(len(y))
x_train, x_test, y_train , y_test =model_selection.train_test_split(x,y,test_size=0.2)

#classifier
clf = LinearRegression()
clf.fit(x_train,y_train)

#pickling 
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)


accuracy = clf.score(x_test,y_test)
forecast_set = clf.predict(x_lately)
print(accuracy, forecast_out,forecast_set)
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix =last_date.timestamp()
one_day=86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date =datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

clf = svm.SVR()
clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test)
forecast_set = clf.predict(x_lately)
print(accuracy, forecast_out,forecast_set)