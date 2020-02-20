#!/usr/bin/python

#Functions Declarations

def calculateRatio(m1, m2):
    ratio =0
    if m1 == "NaN" or m2 == "NaN":
        return ratio
    ratio = m1 / float(m2)
    return ratio

def createNewFeatures(dataset, all_features):
    count = 0
    all_features = all_features + ['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio' ]
    for poi_name in dataset:
        data_point = dataset[poi_name]
        data_point['from_poi_to_this_person_ratio'] = calculateRatio(data_point['from_poi_to_this_person'],data_point['to_messages'])
        data_point['from_this_person_to_poi_ratio'] = calculateRatio(data_point['from_this_person_to_poi'], data_point['from_messages'])
        #print data_point['from_poi_to_this_person_ratio'], data_point['from_this_person_to_poi_ratio' ]
        if count == 0 :
            print data_point
            count = 1
    return dataset, all_features

def selectBestFeatures(dataset, features_list, k):
    data = featureFormat(dataset, features_list)
    labels, features = targetFeatureSplit(data)

    from sklearn.feature_selection import SelectKBest
    selector = SelectKBest(k=k)
    selector.fit_transform(features, labels)
    scores = selector.scores_

    unsorted_list = zip(features_list[1:], scores)

    sorted_list = list(sorted(unsorted_list, key = lambda x: x[1]))

    #print sorted_list
    k_best_features = dict(sorted_list[:k])
    #print k_best_features
    return k_best_features

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi', 'salary'] # You will need to use more features

email_features = ['email_address', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

email_features.remove('email_address') # removed email_address because it will not contribute to the analysis

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

#features_list = email_features + financial_features

features_list = ['poi', 'deferral_payments', 'restricted_stock_deferred', 'shared_receipt_with_poi', 'deferred_income']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

countPOI = 0
countNON = 0

print len(data_dict), ": Before removing outlier"

for i in data_dict:
    for j , k in data_dict[i].items():
        if j == "poi" and k == False: countNON += 1
        elif j == "poi" and k == True: countPOI += 1

print "Total POI: ", countPOI
print "Total Non - POI: ",  countNON

### Task 2: Remove outliers
##We found 2 outliers 1. the row where all the totals are calculated and 2. LOCKHART EUGENE E where all the values are blank

data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

print len(data_dict), ": After removing the outliers"

### Store to my_dataset for easy export below.
my_dataset = data_dict

#print len(my_dataset['SHERRIFF JOHN R'].items()), "before Total"

### Task 3: Create new feature(s)
#x, features_list = createNewFeatures(my_dataset, features_list)
#print len(x['SHERRIFF JOHN R'].items()), "new Total"

data = featureFormat(my_dataset, features_list)
#print features_list
labels, features = targetFeatureSplit(data)

### Extract features and labels from dataset for local testing
numberOfFeatures = 4
newFeatures = selectBestFeatures(my_dataset, features_list, numberOfFeatures)

newFeatures = ['poi'] + newFeatures.keys()

#print features_list
#print newFeatures, "k_best_features"
#print newFeatures[:1]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=0, max_depth=2)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)

print acc, ": ACCURACY"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)


