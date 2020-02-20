#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

#print data_dict
data_dict.pop('TOTAL',0)


data = featureFormat(data_dict, features)

for x in data_dict:

    salary = data_dict[x]['salary']
    bonus = data_dict[x]['bonus']
    if (salary > 1000000 and bonus > 5000000 and salary != "NaN"):
        print (data_dict[x])


### your code below
a, b = 0, 0
for point in data:

    salary = point[0]
    if (salary > 1000000 and bonus > 5000000):
        a = salary

    bonus = point[1]
    if (b < bonus):
        b = bonus
    #print(a, b)
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

