
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

plt.interactive(True)


# with open('iris/iris.data','r') as irisfile:
#     file = csv.reader(irisfile)
#     for row in file:
#         print('.'.join(row))

irisdata = pd.read_csv('iris/iris.data')
irisdata.columns = ['lsepal','wsepal','lpetal','wpetal','class']

# build up a pipeline for data visualization.
#print(np.sum(irisdata['lsepal']))
plt.scatter(irisdata['lsepal'],irisdata['wsepal'])
np.percentile(irisdata['lsepal'],25)
plt.boxplot(irisdata['lsepal'])

irisdata['class'].value_counts()
irisdata.plot(kind = 'scatter',x='lsepal',y = 'wsepal')
sns.jointplot(x='lsepal',y='wsepal',data=irisdata, size=7)
# add class features to plot
sns.FacetGrid(irisdata,hue='class',size=7).map(plt.scatter,'lsepal','wsepal').add_legend()
# view boxplot of an attribute. Add individual points.
sns.boxplot(x='class',y='lsepal',data=irisdata)
sns.stripplot(x='class',y='lsepal',data = irisdata,jitter = True, edgecolor = 'gray')
# feature distributions
sns.FacetGrid(irisdata,hue='class',size=7).map(sns.kdeplot,'lsepal').add_legend()
sns.pairplot(irisdata,hue='class',size=2)
sns.pairplot(irisdata,hue='class',size=2,diag_kind='kde')

# ML part

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train,test = train_test_split(irisdata,test_size=0.3)
trainX = train[['lsepal','wsepal','lpetal','wpetal']]
trainY = train[['class']]
testX = test[['lsepal','wsepal','lpetal','wpetal']]
testY = test[['class']]

# logistic regression

model = LogisticRegression()
model.fit(trainX,trainY)
prediction = model.predict(testX)
metrics.accuracy_score(prediction,testY)