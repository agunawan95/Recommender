import pandas as pd
import time
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("hr.csv")
df.drop(['sales', 'salary'], inplace=True, axis=1)
print df.head(10)

Y = df['left']
X = df.drop('left', axis=1)


def classifier_nb(x, y):
    clf = GaussianNB()
    start = time.clock()
    clf.fit(x, y)
    end = time.clock()

    scores = cross_val_score(clf, x, y, cv=10)
    return scores.mean(), scores.std() * 2, end - start


def classifier_dt(x, y):
    clf = tree.DecisionTreeClassifier()
    start = time.clock()
    clf.fit(x, y)
    end = time.clock()

    scores = cross_val_score(clf, x, y, cv=10)
    return scores.mean(), scores.std() * 2, end - start


def classifier_lr(x, y):
    clf = LogisticRegression()
    start = time.clock()
    clf.fit(x, y)
    end = time.clock()

    scores = cross_val_score(clf, x, y, cv=10)
    return scores.mean(), scores.std() * 2, end - start


classifier_lr(X, Y)
classifier_dt(X, Y)
classifier_nb(X, Y)

pool = ThreadPool(processes=3)

async_result_lr = pool.apply_async(classifier_lr, (X, Y))
async_result_dt = pool.apply_async(classifier_dt, (X, Y))
async_result_nb = pool.apply_async(classifier_nb, (X, Y))

print async_result_lr.get()
print async_result_dt.get()
print async_result_nb.get()
