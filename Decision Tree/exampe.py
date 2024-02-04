import pandas as pd 
from pandas import Series,DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt 

data = {'Age':[36,42,23,52,43,44,66,35,52,35,24,18,45],'Experience':[10,12,4,4,21,14,3,14,13,5,3,3,9],'Rank':[9,4,6,4,8,5,7,9,7,9,5,7,9],'Nationality':['UK','USA','N','USA','USA','UK','N','UK','N','N','USA','UK','UK'],'Go':['NO','NO','NO','NO','YES','NO','YES','YES','YES','YES','NO','YES','YES']}
data_frame = DataFrame(data)

d = {'UK':0, 'USA':1, 'N':2}
data_frame['Nationality'] = data_frame['Nationality'].map(d)
d = {'YES':1,'NO':0}
data_frame['Go'] = data_frame['Go'].map(d)

#print(data_frame)

features = ['Age','Experience','Rank','Nationality']

X = data_frame[features]
y = data_frame['Go']

#print(X)
#print(y)
data_frametree = DecisionTreeClassifier()
data_frametree = data_frametree.fit(X, y)

plt.figure(figsize=(10,10))
tree.plot_tree(data_frametree, feature_names=features, filled = True)
plt.show()

