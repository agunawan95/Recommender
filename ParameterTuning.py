import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class ParametersTuning:

    def dt_parameter_tuning(self, x, y, fold):
        param_grid = {
            "max_depth": [1, 5, 10, 15, 20, 25, 30],
            "min_samples_leaf": [1, 2, 4, 6, 8, 10]
        }
        clf = tree.DecisionTreeClassifier()
        grid_search = GridSearchCV(clf, param_grid, cv=fold)
        grid_search.fit(x, y)
        scores = cross_val_score(grid_search, x, y, cv=fold)
        res = {
            "parameter": grid_search.best_params_,
            "accuracy": grid_search.best_score_,
            "cv_accuracy": scores.mean(),
            "cv_error": scores.std(),
            "cv": scores
        }
        return res


df = pd.read_csv('west_nile_clean.csv')
print df.head(10)
pt = ParametersTuning()

x = df.drop("WnvPresent", axis=1)
y = df['WnvPresent']

param = pt.dt_parameter_tuning(x, y, 10)
print param

