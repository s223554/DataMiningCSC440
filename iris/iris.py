import csv
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

plt.interactive(True)


with open('iris/iris.data','r') as irisfile:
    file = csv.reader(irisfile)
    for row in file:
        print('.'.join(row))

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

