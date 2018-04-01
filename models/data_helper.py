# module data_helper.py

# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# functions

def load_test_case():

    features = pd.read_csv("C:\\Users\\Familj\\PycharmProjects\\Tensor_Flow\\"
                           "datasets\\AAPL_full.csv")

    labels = np.array(features['Close'])

    # Remove the labels from the features
    features = features.drop(['Close'], axis=1)

    # Saving feature names for later use
    feature_list = list(features.columns)

    # Convert to numpy array
    features = np.array(features)

    # Split the data into training and testing sets Using Skicit-learn
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.25, random_state = 42)

    return train_features, test_features, train_labels, test_labels, feature_list