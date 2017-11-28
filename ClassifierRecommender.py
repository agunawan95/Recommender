import pandas as pd
import Recommender
import time
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression


class ClassifierRecommender(Recommender.Recommender):

    def __init__(self):
        Recommender.Recommender.__init__(self)

    def classifier_nb(self, x, y):
        clf = GaussianNB()
        start = time.clock()
        clf.fit(x, y)
        end = time.clock()

        scores = cross_val_score(clf, x, y, cv=10)
        return {
            "name": "Naive Bayes",
            "cv": scores,
            "accuracy": scores.mean(),
            "error": scores.std() * 2,
            "time": end - start
        }

    def classifier_dt(self, x, y):
        clf = tree.DecisionTreeClassifier()
        start = time.clock()
        clf.fit(x, y)
        end = time.clock()

        scores = cross_val_score(clf, x, y, cv=10)
        return {
            "name": "Decision Tree",
            "cv": scores,
            "accuracy": scores.mean(),
            "error": scores.std() * 2,
            "time": end - start
        }

    def classifier_lr(self, x, y):
        clf = LogisticRegression()
        start = time.clock()
        clf.fit(x, y)
        end = time.clock()

        scores = cross_val_score(clf, x, y, cv=10)
        return {
            "name": "Logistic Regression",
            "cv": scores,
            "accuracy": scores.mean(),
            "error": scores.std() * 2,
            "time": end - start
        }

    def run(self):
        pool = ThreadPool(processes=3)
        self.result.append(pool.apply_async(self.classifier_lr, (self.x, self.y)).get())
        self.result.append(pool.apply_async(self.classifier_dt, (self.x, self.y)).get())
        self.result.append(pool.apply_async(self.classifier_nb, (self.x, self.y)).get())



