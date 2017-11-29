import pandas as pd
import Recommender
import time
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LassoCV


class RegressorRecommender(Recommender.Recommender):

    def __init__(self):
        Recommender.Recommender.__init__(self)

    def regressor_rt(self, x, y):
        regressor = tree.DecisionTreeClassifier()
        start = time.clock()
        regressor.fit(x, y)
        end = time.clock()

        scores = np.sqrt(-cross_val_score(regressor, x, y, scoring="neg_mean_squared_error", cv=self.get_cross_validation_split()))
        return {
            "name": "Regression Tree",
            "cv": scores,
            "accuracy": scores.mean(),
            "error": scores.std() * 2,
            "time": end - start
        }

    def regressor_svr(self, x, y):
        regressor = svm.SVR()
        start = time.clock()
        regressor.fit(x, y)
        end = time.clock()

        scores = np.sqrt(-cross_val_score(regressor, x, y, scoring="neg_mean_squared_error", cv=self.get_cross_validation_split()))
        return {
            "name": "SVR",
            "cv": scores,
            "accuracy": scores.mean(),
            "error": scores.std() * 2,
            "time": end - start
        }

    def regressor_lasso(self, x, y):
        regressor = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(x, y)
        start = time.clock()
        regressor.fit(x, y)
        end = time.clock()

        scores = np.sqrt(-cross_val_score(regressor, x, y, scoring="neg_mean_squared_error", cv=self.get_cross_validation_split()))
        return {
            "name": "Lasso Regression",
            "cv": scores,
            "accuracy": scores.mean(),
            "error": scores.std() * 2,
            "time": end - start
        }

    def run(self):
        pool = ThreadPool(processes=3)
        self.result.append(pool.apply_async(self.regressor_rt, (self.x, self.y)).get())
        self.result.append(pool.apply_async(self.regressor_svr, (self.x, self.y)).get())
        self.result.append(pool.apply_async(self.regressor_lasso, (self.x, self.y)).get())
