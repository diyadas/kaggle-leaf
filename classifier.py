import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from sklearn.pipeline import make_pipeline
from sklearn import svm, neighbors, ensemble, preprocessing
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
#Build an optimizer for the best RF
def optimized_classifier(X, y, classifier, distributions, scorer='f1_weighted', n_iter=50, cv=3):
    """
    Return best classifier and scores for X,y from a randomized search over parameters

    X             -- Features for each sample
    y             -- Class label for each sample
    classifier    -- An estimator class or pipeline from sklearn
    distributions -- The parameter distributions to search for that estimator
    scorer        -- Scoring function (e.g. accuracy or f1)
    n_iter        -- The number of random iterations to try
    """
    # Make a pipeline out of the classifier, to allow for feature scaling in the first step.

    # Add prefix to parameters to support use in pipeline
    class_name = classifier.__class__.__name__.lower()
    distributions = dict((class_name + "__" + key, val) for key, val in distributions.iteritems())

    # It is important to handle scaling here so we don't accidentally overfit some to the
    # test data by scaling using that information as well.
    classifier = make_pipeline(preprocessing.RobustScaler(), classifier)
    randomized_search = RandomizedSearchCV(
        classifier, param_distributions=distributions, n_iter=n_iter, scoring=scorer, cv=cv, n_jobs=1)
    randomized_search.fit(X, y)

    print randomized_search.best_estimator_
    print "Validation Score ({}): {:.2f}".format(scorer, randomized_search.best_score_)
    print ""
    return [randomized_search.best_estimator_, randomized_search.best_score_]
