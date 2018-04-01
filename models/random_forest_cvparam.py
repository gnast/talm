# module random_forest_cvparam.py

# imports
from __future__ import print_function
from __future__ import division
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from data_helper import load_test_case
import pandas as pd
import time
import csv

# functions
def print_shapes():
    print("Training Features Shape:", train_features.shape)
    print("Training Labels Shape:", train_labels.shape)
    print("Test Features Shape:", test_features.shape)
    print("Test Labels Shape:", test_labels.shape)

def save_cv_param(param_grid):
    w = csv.writer(open("cv_param.csv", "w"))
    for key, val in param_grid.items():
        w.writerow([key, val])
    print("Parameters successfully saved.")


if __name__ == "__main__":
    pd.set_option('expand_frame_repr', False)

    start_time = time.time()

    # Load from data_helper
    train_features, test_features, train_labels, test_labels, feature_list = load_test_case()

    # Instantiate model with 1000 decision trees, set n_jobs = -1 = number of cores for Cross Validation optimization
    #rfc = RandomForestRegressor(n_estimators=3000)

    rfc = GradientBoostingRegressor(n_estimators=3000)

    # Values to test CV-optimize Gradient-Boosting
    param_grid_gb = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'max_features': [0.1, 'auto', 'sqrt'],
        'max_depth': [4, 5, 6],
        'min_samples_leaf': [3,5,7]
    }

    # Values to test CV-optimize Regressor
    param_grid_reg = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [5, 7, 10],
        'min_samples_leaf': [40, 50, 60]
    }

    # Run cv-optimizer baed on grid and fit for
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_gb, n_jobs=4).fit(train_features, train_labels)

    save_cv_param(CV_rfc.best_params_)

    print("Duration in Seconds:", (time.time() - start_time))