#!/usr/bin/python

import pickle
import sys
from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tester import test_classifier, dump_classifier_and_data

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options',
                 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred',
                 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other',
                 'from_this_person_to_poi',
                 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


def deleteOutliers(dictionary, financial_data, contamination=0.02):
    financial_labels, financlial_features = targetFeatureSplit(financial_data)
    clf = LocalOutlierFactor(contamination=contamination)
    print("IsolationForest")
    for key, out in zip(dictionary.keys(), clf.fit_predict(financlial_features, financial_labels)):
        if (out == -1):
            print("REMOVING: %s: " % key)
            del dictionary[key]
    return dictionary


def getFeatures(data_dict, features_list, k=10):
    ### Task 2: Remove outliers
    del data_dict["TOTAL"]
    del data_dict["LOCKHART EUGENE E"]
    del data_dict["THE TRAVEL AGENCY IN THE PARK"]
    # data_dict = deleteOutliers(data_dict, featureFormat(data_dict, features_list, sort_keys=True))

    ### Task 3: Create new feature(s)
    new_features = []
    for index, (key, value) in enumerate(data_dict.iteritems()):
        tmp1 = -1
        tmp2 = -1
        tmp3 = -1
        tmp4 = -1
        for k1, v1 in value.iteritems():
            if k1 == 'from_messages':
                if str(v1) != v1:
                    tmp1 = v1
            if k1 == 'from_poi_to_this_person':
                if str(v1) != v1:
                    tmp2 = v1
            if k1 == 'to_messages':
                if str(v1) != v1:
                    tmp3 = v1
            if k1 == 'from_this_person_to_poi':
                if str(v1) != v1:
                    tmp4 = v1
        if tmp1 != -1 and tmp2 != -1:
            data_dict[key]['from_poi_ratio'] = float(tmp2) / float(tmp1)
        else:
            data_dict[key]['from_poi_ratio'] = 'NaN'
        if tmp3 != -1 and tmp4 != -1:
            data_dict[key]['to_poi_ratio'] = float(tmp4) / float(tmp3)
        else:
            data_dict[key]['to_poi_ratio'] = 'NaN'

        t0 = data_dict[key]['poi']
        t1 = 0
        t2 = 0
        for k1, v1 in value.iteritems():
            if k1 == 'from_poi_ratio':
                if str(v1) != v1: t1 = v1
            if k1 == 'to_poi_ratio':
                if str(v1) != v1: t2 = v1
        new_features.append((t0, t1, t2))

    features_list.append('from_poi_ratio')
    features_list.append('to_poi_ratio')

    ### Choose features using SelectKBest
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42, stratify=labels)
    ### Scale features
    scaler = MinMaxScaler()
    features_train_scaled = scaler.fit_transform(features_train)

    pca = PCA(n_components=10)
    features_train_pca = pca.fit_transform(features_train_scaled)

    k_best = SelectKBest(k=k)
    # k_best.fit(features_train_scaled, labels_train)
    k_best.fit(features_train_pca, labels_train)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = OrderedDict(sorted_pairs[:k])

    print "{0} best features: {1} and score {2}\n".format(k, k_best_features.keys(), k_best_features.values())

    return k_best_features.keys()


feature_selection = getFeatures(data_dict, features_list, 7)
selected_features_list = ['poi'] + feature_selection

data = featureFormat(data_dict, selected_features_list)
labels, features = targetFeatureSplit(data)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                    random_state=42, stratify=labels)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# clf = DecisionTreeClassifier(min_samples_split=100)
# clf = SVC(C=10.0, gamma=0.001)
# clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=100), algorithm="SAMME")
clf = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="auto")

test_classifier(clf, data_dict, selected_features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# # tuned_parameters = {"criterion": ("gini", "entropy"), "max_depth": (None, 1, 2, 5, 7, 10), "min_samples_split": (10, 100, 250)}
# # tuned_parameters = {"C": (10.0, 100.0, 1000.0), "gamma": (1e-3, 1e-4)}
# # tuned_parameters = {"n_estimators": (50, 100, 150, 200), "learning_rate": (1.0, 1.5, 2.0), "algorithm": ("SAMME", "SAMME.R")}
# tuned_parameters = {"n_neighbors": (1, 5, 10, 15), "weights": ("uniform", "distance")}
#
# gs = GridSearchCV(clf, tuned_parameters, cv=10)
# gs.fit(X_train, y_train)
#
# y_true, y_pred = y_test, gs.predict(X_test)
#
# print "Best params are {0}, with score: {1}.".format(gs.best_params_, gs.best_score_)
# print("Detailed classification report:")
# print()
# print("The model is trained on the full development set.")
# print("The scores are computed on the full evaluation set.")
# print()
# print(classification_report(y_true, y_pred))
# print()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, data_dict, selected_features_list)