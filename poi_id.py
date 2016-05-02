#!/usr/bin/python

import sys
import pickle
import numpy as np
import json
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pprint
pp = pprint.PrettyPrinter(indent=4)

# Algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Dimensionality Reduction
from sklearn.decomposition import RandomizedPCA

# Feature selection
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

# Pipelines
from sklearn.pipeline import Pipeline

# Parameter tuning
from sklearn.grid_search import GridSearchCV

# Testing and validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report


#################
### Load Data ###
#################

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



#######################
### Remove outliers ###
#######################

# Data glitch
data_dict.pop('TOTAL', 0)

# Not a person
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# Has no data
data_dict.pop('LOCKHART EUGENE E', 0)



########################
### Add New Features ###
########################

### Total POI Emails 
for person in data_dict.keys():
    if data_dict[person]['from_poi_to_this_person'] == 'NaN' or \
        data_dict[person]['from_this_person_to_poi'] == 'NaN' or \
        data_dict[person]['shared_receipt_with_poi'] == 'NaN':
        data_dict[person]['total_poi_emails'] = 'NaN'
    else:
        data_dict[person]['total_poi_emails'] = \
            data_dict[person]['from_poi_to_this_person'] + \
            data_dict[person]['from_this_person_to_poi'] + \
            data_dict[person]['shared_receipt_with_poi']

### Fraction of POI EMails 
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = float(poi_messages) / float(all_messages)
    else:
        return 0.

    return fraction

for person in data_dict:
    # from POI to person
    from_poi_to_this_person = data_dict[person]['from_poi_to_this_person']
    to_messages = data_dict[person]['to_messages']
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_dict[person]['fraction_from_poi'] = fraction_from_poi

    # from person to POI
    from_this_person_to_poi = data_dict[person]['from_this_person_to_poi']
    from_messages = data_dict[person]['from_messages']
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_dict[person]['fraction_to_poi'] = fraction_to_poi



################################
### Manual Feature Selection ###
################################

# Note: 'email_adress' excluded in all feature lists

### All original features
# features_list = ['poi','loan_advances','director_fees','restricted_stock_deferred','deferral_payments','deferred_income','long_term_incentive','bonus', 'from_messages','from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi','to_messages','other','expenses','salary','exercised_stock_options','restricted_stock','total_payments','total_stock_value']

### All features including newly created
# features_list = ['poi','loan_advances','director_fees','restricted_stock_deferred','deferral_payments','deferred_income','long_term_incentive','bonus','from_messages','from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi','to_messages','other','expenses','salary','exercised_stock_options','restricted_stock','total_payments','total_stock_value','total_poi_emails', 'fraction_from_poi', 'fraction_to_poi']

### Top correlating features to 'poi'
features_list = ['poi','loan_advances','exercised_stock_options','bonus', 'salary']

### Top correlating features with our newly created
# features_list = ['poi', 'loan_advances', 'exercised_stock_options', 'fraction_to_poi', 'bonus', 'salary']



###################################
### Data Load for Local Testing ###
###################################

### Extract features and labels from dataset for local testing
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



#########################
### Pipelines to Test ###
#########################

### Estimator parameters

random_state = [42]

# Decision Trees
criterion = ['gini']
splitter = ['best']
max_features = [None,]
min_samples_split = [9]

# Further Decision Tree fine-tuning (once it was chosen)
max_depth = [None]
min_samples_leaf = [1]
max_leaf_nodes = [None]
presort = [True]


# Random Forests
n_estimators = [4, 6]

# SVC
C = [1000]
kernel = ['sigmoid', 'rbf', 'poly']
degree = [3]
gamma = [1, 2, 3]

# KMeans
n_clusters = [2]

# PCA
n_components = [2]
whiten = [False]

# SelectKBest
k = range(2, len(features_list), 3)

### Naive Bayes ###

# Stand-alone
# estimators = [('nb', GaussianNB())]
# param_grid = {}

# With PCA
# estimators = [('pca', RandomizedPCA()), ('nb', GaussianNB())]
# param_grid = {'pca__n_components': n_components, 'pca__whiten': whiten, 'pca__random_state': random_state}

# With SelectKbest
# estimators = [('kbest', SelectKBest()), ('nb', GaussianNB())]
# param_grid = {'kbest__k': k}


### Decision Tree ###

# Stand-alone
# estimators = [('tree', DecisionTreeClassifier())]
# param_grid = {'tree__criterion': criterion, 'tree__splitter': splitter, 'tree__max_features': max_features, 'tree__min_samples_split': min_samples_split}

# With PCA
estimators = [('pca', RandomizedPCA()), ('tree', DecisionTreeClassifier())]
param_grid = {'pca__n_components': n_components, 'pca__whiten': whiten, 'pca__random_state': random_state,
                'tree__criterion': criterion, 'tree__splitter': splitter, 'tree__max_features': max_features, 'tree__min_samples_split': min_samples_split, 'tree__random_state': random_state,
                'tree__max_depth': max_depth, 'tree__min_samples_leaf': min_samples_leaf, 'tree__max_leaf_nodes': max_leaf_nodes, 'tree__presort': presort}

# With SelectKbest
# estimators = [('kbest', SelectKBest()), ('tree', DecisionTreeClassifier())]
# param_grid = {'kbest__k': k,
#               'tree__criterion': criterion, 'tree__splitter': splitter, 'tree__max_features': max_features, 'tree__min_samples_split': min_samples_split}


### Random Forest ###

# Stand-alone
# estimators = [('tree', RandomForestClassifier())]
# param_grid = {'tree__criterion': criterion, 'tree__max_features': max_features, 'tree__min_samples_split': min_samples_split, 'tree__n_estimators': n_estimators}

# With PCA
# estimators = [('pca', RandomizedPCA()), ('tree', RandomForestClassifier())]
# param_grid = {'pca__n_components': n_components, 'pca__whiten': whiten, 'pca__random_state': random_state,
#              'tree__criterion': criterion, 'tree__max_features': max_features, 'tree__min_samples_split': min_samples_split, 'tree__n_estimators': n_estimators}

# With SelectKbest
# estimators = [('kbest', SelectKBest()), ('tree', RandomForestClassifier())]
# param_grid = {'kbest__k': k,
#               'tree__criterion': criterion, 'tree__max_features': max_features, 'tree__min_samples_split': min_samples_split, 'tree__n_estimators': n_estimators}


### SVM ###

# Stand-alone
# estimators = [('svm', SVC())]
# param_grid = {'svm__C': C, 'svm__kernel': kernel, 'svm__degree': degree, 'svm__gamma': gamma}

# With PCA
# estimators = [('pca', RandomizedPCA()), ('svm', SVC())]
# param_grid = {'pca__n_components': n_components, 'pca__whiten': whiten, 'pca__random_state': random_state,
#              'svm__C': C, 'svm__kernel': kernel, 'svm__degree': degree, 'svm__gamma': gamma}

# With PCA and Scaling
# estimators = [('pca', RandomizedPCA()), ('scale', MinMaxScaler()), ('svm', SVC())]
# param_grid = {'pca__n_components': n_components, 'pca__whiten': whiten, 'pca__random_state': random_state,
#              'svm__C': C, 'svm__kernel': kernel, 'svm__degree': degree, 'svm__gamma': gamma}

# With Scaling
# estimators = [('scale', MinMaxScaler()), ('svm', SVC())]
# param_grid = {'svm__C': C, 'svm__kernel': kernel, 'svm__degree': degree, 'svm__gamma': gamma}

# With Scaling and SelectKest
# estimators = [('scale', MinMaxScaler()), ('kbest', SelectKBest()), ('svm', SVC())]
# param_grid = {'kbest__k': k,
#              'svm__C': C, 'svm__kernel': kernel, 'svm__degree': degree, 'svm__gamma': gamma}

# With SelectKbest
# estimators = [('kbest', SelectKBest()), ('svm', SVC())]
# param_grid = {'kbest__k': k,
#              'svm__C': C, 'svm__kernel': kernel, 'svm__degree': degree, 'svm__gamma': gamma}


### KMeans ###

# Stand-alone
# estimators = [('kmeans', KMeans())]
# param_grid = {'kmeans__n_clusters': n_clusters, 'kmeans__random_state': random_state}

# With PCA
# estimators = [('pca', RandomizedPCA()), ('kmeans', KMeans())]
# param_grid = {'pca__n_components': n_components, 'pca__whiten': whiten, 'pca__random_state': random_state,
#              'kmeans__n_clusters': n_clusters, 'kmeans__random_state': random_state}

# With PCA and Scaling
# estimators = [('pca', RandomizedPCA()), ('scale', MinMaxScaler()), ('kmeans', KMeans())]
# param_grid = {'pca__n_components': n_components, 'pca__whiten': whiten, 'pca__random_state': random_state,
#              'kmeans__n_clusters': n_clusters, 'kmeans__random_state': random_state}

# With Scaling
# estimators = [('scale', MinMaxScaler()), ('kmeans', KMeans())]
# param_grid = {'kmeans__n_clusters': n_clusters, 'kmeans__random_state': random_state}

# With Scaling and SelectKest
# estimators = [('scale', MinMaxScaler()), ('kbest', SelectKBest()), ('kmeans', KMeans())]
# param_grid = {'kbest__k': k,
#              'kmeans__n_clusters': n_clusters, 'kmeans__random_state': random_state}

# With SelectKbest
# estimators = [('kbest', SelectKBest()), ('kmeans', KMeans())]
# param_grid = {'kbest__k': k,
#              'kmeans__n_clusters': n_clusters, 'kmeans__random_state': random_state}


### Build Pipeline
pipe = Pipeline(estimators)

### Set up Grid
clf = GridSearchCV(pipe, param_grid, scoring = 'recall')


###############################################
### And the winner is... DecisionTree + PCA ###
###############################################

### In order to get the feature importance for the assignment, I will rebuild the pipe step by step to output the importances
grid = True # Set to false to run manual classifier below
if not grid:    
    pca = RandomizedPCA(n_components=2, random_state=random_state[0])
    features = pca.fit_transform(np.array(features))
    print "PCA Explained Variance: {}".format(pca.explained_variance_ratio_)

    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_features=None, min_samples_split=9, random_state=random_state[0])
    grid = False



#################################
### Validation and Evaluation ###
#################################

### Evaluation Output Functions
def get_prediction_results(predictions, labels_test, 
                           true_negatives, false_negatives,
                           true_positives, false_positives):
    # From test.py
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predictebd label not == 0 or 1: pred: {}, truth{}".format(prediction, truth)
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break

    return true_negatives, false_negatives, true_positives, false_positives

def print_summary(clf, true_negatives, false_negatives, true_positives, false_positives):
    # From test.py
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        
        print '\nAcc: {:.2f} | Prec: {:.2f} | Recall: {:.2f} | F1: {:.2f} | F2: {:.2f}'.format(accuracy, precision, recall, f1, f2)

    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

def print_best_params(best_params):
    best_params = [json.dumps(d, sort_keys=True) for d in best_params]
    param_counts = []
    for param in set(best_params):
        param_counts.append([param, best_params.count(param)])
    print '### GridSearchCV Best Parameters'
    for param in sorted(param_counts, key=lambda x: x[1], reverse=True):
        print param[1], param[0]


### Standard one-time train/test split
def one_time_split_validation(clf, features, labels, grid=True):
    # Output results
    print '\n\n### Standard one-time train/test split'

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    true_negatives, false_negatives, true_positives, false_positives = [0] * 4
    true_negatives, false_negatives, true_positives, false_positives = get_prediction_results(predictions, labels_test, 
                                                                                                  true_negatives, false_negatives,
                                                                                                  true_positives, false_positives)

    print_summary(clf, true_negatives, false_negatives, true_positives, false_positives)
    if grid: 
        print_best_params([clf.best_params_])
    else:
        # Output things about a specific classifier
        print 'Decision Tree feature importances: {}'.format(clf.feature_importances_)

# one_time_split_validation(clf, features, labels, grid=grid)


### KFold Validation
def kfold_validation(clf, features, labels, grid=True):
    kf = KFold(len(features), n_folds=2)

    true_negatives, false_negatives, true_positives, false_positives = [0] * 4
    best_params = []

    # Output results
    print '\n\n### KFold Validation'

    for train_index, test_index in kf:
        # Split
        features_train, features_test = np.asarray(features)[train_index], np.asarray(features)[test_index]
        labels_train, labels_test = np.asarray(labels)[train_index], np.asarray(labels)[test_index]

        # Fit
        clf.fit(features_train, labels_train)

        # Predict
        predictions = clf.predict(features_test)

        # Get Grid params
        if grid:
            best_params.append(clf.best_params_)

        true_negatives, false_negatives, true_positives, false_positives = get_prediction_results(predictions, labels_test, 
                                                                                                  true_negatives, false_negatives,
                                                                                                  true_positives, false_positives)

        print_summary(clf, true_negatives, false_negatives, true_positives, false_positives)
        print_best_params(best_params)

# kfold_validation(clf, features, labels, grid=grid)

### StratifiedShuffleSplit (from tester.py)
def stratified_validation(clf, features, labels, folds = 100, grid=True):
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    
    true_negatives, false_negatives, true_positives, false_positives = [0] * 4
    i = 0 #
    best_params = [] #
    
    print '\n\n### Stratified Validation'
    
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        if i % 10 == 0: print 'Processing fold {} to {}...'.format(i+1, i+10)
        i += 1
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        if grid:
            best_params.append(clf.best_params_)
        true_negatives, false_negatives, true_positives, false_positives = get_prediction_results(predictions, labels_test, 
                                                                                                  true_negatives, false_negatives,
                                                                                                  true_positives, false_positives)

    print_summary(clf, true_negatives, false_negatives, true_positives, false_positives)
    print_best_params(best_params)

stratified_validation(clf, features, labels, grid=grid)


#######################
### Dump for Tester ###
#######################

dump_classifier_and_data(clf, my_dataset, features_list)





