import csv
import pandas as pd
import numpy as np
import matplotlib

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