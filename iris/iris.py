import csv
import pandas as pd
import numpy as np

with open('iris.data','r') as irisfile:
    file = csv.reader(irisfile)
    for row in file:
        print('.'.join(row))

irisdata = pd.read_csv('iris.data')
irisdata.columns = ['lsepal','wsepal','lpetal','wpetal','class']

# build up a pipeline for data visualization.
print(np.sum(irisdata['lsepal']))
