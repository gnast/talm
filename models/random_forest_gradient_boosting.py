# module random_forest_cvparam.py

# imports
from __future__ import print_function
from __future__ import division
from sklearn.ensemble import GradientBoostingRegressor
from data_helper import load_test_case
import pandas as pd
import numpy as np
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
    rf = GradientBoostingRegressor(loss='ls', learning_rate=0.05, max_features='sqrt', n_estimators=700,
                                   random_state=42, max_depth=5, min_samples_leaf=9)

    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)
    # print(predictions)

    # Calculate the absolute erros
    errors = abs(predictions - test_labels)
    print("Errors f(pred - test):", errors)

    # Print out the mean absolute error (mae)
    print("Mean Absolute Error:", round(np.mean(errors), 2), "degrees.")

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    print(mape)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)

    print("Duration in Seconds:", (time.time() - start_time))