#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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


# Train on 1% of the data to speed it up
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###

from sklearn import svm
from sklearn.metrics import accuracy_score
from timing import time_function

def linear_svm_classify(features_train, labels_train, features_test, labels_test):
    '''
    Perform linear SVM classification on the data.
    '''
    linear_SVM = svm.SVC(kernel='linear')
    print('Using kernel {}'.format(linear_SVM.kernel))

    time_function(lambda: linear_SVM.fit(features_train, labels_train), 'linear_SVM.fit()')

    labels_predicted = time_function(lambda: linear_SVM.predict(features_test), 'linear_SVM.predict()')

    accuracy = time_function(lambda: accuracy_score(labels_predicted, labels_test), 'accuracy_score()')

    print('Linear accuracy: {}'.format(accuracy))

def get_best_c_value_rbf(features_train, labels_train, features_test, labels_test, c_values):
    '''
    From a given set of c values, determine the one that yields the best accuracy.
    '''
    best_c_value = -1.0
    best_accuracy = 0.0
    for c in c_values:
        rbf_SVM = svm.SVC(kernel='rbf', C=c, gamma='auto')
        print('Using kernel {}'.format(rbf_SVM.kernel))

        time_function(lambda: rbf_SVM.fit(features_train, labels_train), 'rbf_SVM.fit()')

        labels_predicted = time_function(lambda: rbf_SVM.predict(features_test), 'rbf_SVM.predict()')

        accuracy = time_function(lambda: accuracy_score(labels_predicted, labels_test), 'accuracy_score()')

        if accuracy > best_accuracy:
            best_c_value = c

        print('RBF accuracy with c={}: {}'.format(c, accuracy))

    return best_c_value

best_c_value = get_best_c_value_rbf(features_train[:len(features_train)/100],
                                    labels_train[:len(labels_train)/100],
                                    features_test,
                                    labels_test,
                                    [10.0, 100.0, 1000.0, 10000.0])

print('Got best C value: {}'.format(best_c_value))

def rbf_classify_with_c(features_train, labels_train, features_test, labels_test, c):
    '''
    Perform SVM classification with rbf kernel, using an optimal c parameter passed in.
    '''
    rbf_SVM = svm.SVC(kernel='rbf', C=c, gamma='auto')
    print('Using kernel {}'.format(rbf_SVM.kernel))

    time_function(lambda: rbf_SVM.fit(features_train, labels_train), 'rbf_SVM.fit()')
    
    labels_predicted = time_function(lambda: rbf_SVM.predict(features_test), 'rbf_SVM.predict()')
    
    accuracy = time_function(lambda: accuracy_score(labels_predicted, labels_test), 'accuracy_score()')

    print('RBF accuracy with c={}: {}'.format(c, accuracy))


rbf_classify_with_c(features_train, labels_train, features_test, labels_test, best_c_value)


#########################################################


