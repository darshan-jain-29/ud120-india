#!/usr/bin/python
from __builtin__ import sorted


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    #print (predictions, "Predictions")
    #print (net_worths, "Paisaaaaaaaaaa")
    cleaned_data = []

    ### your code goes here
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    print (r2_score(net_worths, predictions, multioutput='variance_weighted'))
    errors = []
    for xNet, xPred in zip(net_worths, predictions):
        errors.append(mean_squared_error(xNet, xPred))

    tempList = zip(ages, net_worths, errors)
    #print (tempList)

    sortedList = sorted(tempList, key=lambda x: x[2])
    #print (sortedList)

    cleaned_data = sortedList[:81]
    return cleaned_data

