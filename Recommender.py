import pandas as pd


def multikeysort(items, columns, reverse):
    from operator import itemgetter
    comparers = [((itemgetter(col[1:].strip()), -1) if col.startswith('-') else
                  (itemgetter(col.strip()), 1)) for col in columns]

    def comparer(left, right):
        for fn, mult in comparers:
            result = cmp(fn(left), fn(right))
            if result:
                return mult * result
        else:
            return 0
    return sorted(items, cmp=comparer, reverse=reverse)


class Recommender:
    df = None
    x = None
    y = None
    result = []

    def __init__(self):
        self.df = None
        self.x = None
        self.y = None
        self.result = []

    def set_data(self, df):
        self.df = df

    def define_target(self, target):
        self.y = self.df[target]
        self.x = self.df.drop(target, axis=1)

    def reset(self):
        self.df = None
        self.x = None
        self.y = None
        self.result = []

    def get_cross_validation_split(self):
        if self.df.shape[0] <= 999:
            return 2
        else:
            return 10

    def sort(self, based):
        res = []
        if based == 'accuracy':
            res = multikeysort(self.result, ['accuracy', 'error'], True)
        elif based == 'rmse':
            res = multikeysort(self.result, ['accuracy', 'error'], False)
        elif based == 'time':
            res = multikeysort(self.result, ['time', 'accuracy'], False)
        return res
