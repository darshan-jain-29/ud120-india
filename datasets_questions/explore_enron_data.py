#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_names =  pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

#print enron_names

print enron_names["SKILLING JEFFREY K"]

countSalary = 0
countEmail = 0
for x in enron_names:
    if enron_names[x]["total_payments"]  == "NaN" and enron_names[x]["poi"]  == True:
        countSalary += 1
    if enron_names[x]["total_payments"] and enron_names[x]["poi"]  == True:
        countEmail += 1
countSalary += 10
countEmail += 10
print countSalary, countEmail

print (countSalary * 100)/countEmail
#data =  enron_names.read()
#with  open('../final_project/poi_names.txt) as f:
 #   print sum(1 for _ in enron_names)


#print ("POI is there in : ", data[0][0])


#print len(enron_data[enron_data.keys()[0]])

