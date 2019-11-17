#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from timing import time_function
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

num_features = len(features_train[0])
print('Num features: {}'.format(num_features))

dt_classifier = DecisionTreeClassifier(min_samples_split=40)

time_function(lambda: dt_classifier.fit(features_train, labels_train), 'dt_classifier.fit()')

predictions = time_function(lambda: dt_classifier.predict(features_test), 'dt_classifier.predict()')

# 0.980091012514, too high for udacity??
accuracy_score_accuracy = time_function(lambda: accuracy_score(predictions, labels_test), 'accuracy_score()')
# mean accuracy: 0.977815699659
accuracy = time_function(lambda: dt_classifier.score(features_test, labels_test), 'dt_classifier.score()')

custom_accuracy = time_function(lambda: (sum([1 for (x, y) in zip(predictions, labels_test) if x == y])*1.0)/len(labels_test), 'custom acc computation')

print('Accuracy_score accuracy: {}'.format(accuracy_score_accuracy))
print('Accuracy: {}'.format(accuracy))
print('Custom accuracy: {}'.format(custom_accuracy))

#########################################################


