import numpy as np
import pandas as pd
import copy
from typing import Union
from sklearn.base import BaseEstimator
import rpy2.robjects as r_objects
from rpy2.robjects import pandas2ri


class IsolationForestUnsupervised(BaseEstimator):

    def __init__(self):
        self.model = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        # assert isinstance(X, pd.DataFrame)

        pandas2ri.activate()

        X_fit = copy.deepcopy(X)

        """
        if isinstance(X, np.ndarray):
            X_fit = np.c_[X_fit, y]
        else:
            X_fit["target"] = y
        """

        X_fit = pd.DataFrame(X_fit)
        X_fit.columns = ['c' + str(i) for i in range(X_fit.shape[1])]

        data_set = pandas2ri.py2rpy_pandasdataframe(X_fit)

        r_objects.r('''
                   train_if <- function(train) {
                        sample_size = 0.1 * nrow(train)

                        library("isofor")
                        
                        model<-iForest(train,
                                nt = 25,
                                phi = ceiling(sample_size))
                                
                        model
                    }
                    ''')

        fit_model = r_objects.globalenv['train_if']
        self.model = fit_model(data_set)

        pandas2ri.deactivate()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        pandas2ri.activate()
        r_objects.r('''
                   predict_if <- function(model,test) {
                            library("isofor")

                            y_hat_probs <- predict(model, test)
                            
                            y_hat_probs
                    }
                    ''')

        predict_method = r_objects.globalenv['predict_if']

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            X.columns = ['c' + str(i) for i in range(X.shape[1])]
        else:
            X.columns = ['c' + str(i) for i in range(X.shape[1])]

        test_data_set = pandas2ri.py2rpy_pandasdataframe(X)

        y_hat = predict_method(self.model, test_data_set)
        pandas2ri.deactivate()

        y_hat_f = (y_hat > 0.5).astype(int)

        return y_hat_f

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        pandas2ri.activate()
        r_objects.r('''
                   predict_if <- function(model,test) {
                            library("isofor")

                            y_hat_probs <- predict(model, test)

                            y_hat_probs
                    }
                    ''')

        predict_method = r_objects.globalenv['predict_if']

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            X.columns = ['c' + str(i) for i in range(X.shape[1])]
        else:
            X.columns = ['c' + str(i) for i in range(X.shape[1])]

        test_data_set = pandas2ri.py2rpy_pandasdataframe(X)

        y_hat = predict_method(self.model, test_data_set)
        pandas2ri.deactivate()

        return y_hat
