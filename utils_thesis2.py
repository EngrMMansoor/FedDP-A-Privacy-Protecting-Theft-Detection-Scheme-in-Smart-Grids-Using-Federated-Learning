from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import openml
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: VotingClassifier) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.flatten_transform:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params


def set_model_params(
    model: VotingClassifier, params: LogRegParams
) -> VotingClassifier:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.flatten_transform:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: VotingClassifier):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 1035  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.flatten_transform:
        model.intercept_ = np.zeros((n_classes,))

# def get_model_parameters(model: LogisticRegression) -> LogRegParams:
#     """Returns the paramters of a sklearn LogisticRegression model."""
#     if model.fit_intercept:
#         params = (model.coef_, model.intercept_)
#     else:
#         params = (model.coef_,)
#     return params


# def set_model_params(
#     model: LogisticRegression, params: LogRegParams
# ) -> LogisticRegression:
#     """Sets the parameters of a sklean LogisticRegression model."""
#     model.coef_ = params[0]
#     if model.fit_intercept:
#         model.intercept_ = params[1]
#     return model


# def set_initial_params(model: LogisticRegression):
#     """Sets initial parameters as zeros Required since model params are
#     uninitialized until model.fit is called.
#     But server asks for initial parameters from clients at launch. Refer
#     to sklearn.linear_model.LogisticRegression documentation for more
#     information.
#     """
#     n_classes = 2  # MNIST has 10 classes
#     n_features = 1034  # Number of features in dataset
#     model.classes_ = np.array([i for i in range(2)])

#     model.coef_ = np.zeros((n_classes, n_features))
#     if model.fit_intercept:
#         model.intercept_ = np.zeros((n_classes,))

def load_nsl_kdd():

    # X_train, X_test, y_train, y_test = train_test_split(df_filldata,label, test_size=0.2, random_state=10) 
    # alldata = pd.read_csv(r"C:\Users\Fatima\Desktop\thesis data\electricity data\china\normalized_authors_data.csv")    
    # authors_label = alldata["flag"]
    # authors_data =alldata.drop(['flag'],axis = 1)
    # # scaler=MinMaxScaler()
    # # scaled_data=scaler.fit_transform(authors_data)
    # # df_filldata = pd.DataFrame(scaled_data)
    # X_train, X_test, y_train, y_test = train_test_split(authors_data,authors_label, test_size=0.4, random_state=10)
    alldata = pd.read_csv(r"C:\Users\Fatima\Desktop\thesis data\electricity data\china\AllData.csv")    
    authors_label = alldata["IsStealer"]
    authors_data =alldata.drop(['IsStealer','UserId'],axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(authors_data,authors_label, test_size=0.2, random_state=75) 
   
    return (X_train, y_train), (X_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
