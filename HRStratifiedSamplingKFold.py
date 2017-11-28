import pandas as pd
import ClassifierRecommender as cr
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

rec = cr.ClassifierRecommender()

df = pd.read_csv("hr.csv")
df.drop(['sales', 'salary'], inplace=True, axis=1)

rec.set_data(df.copy())
rec.define_target('left')
rec.run()
res = rec.sort('accuracy')
for value in res:
    print value['name']

print("---")
x = df.copy()
y = df['left']


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


for p in frange(0.1, 0.6, 0.1):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    rec.reset()
    rec.set_data(x_test.copy())
    rec.define_target('left')
    rec.run()
    result = rec.sort('accuracy')
    print "Percent: " + str(p) + ", Rows: " + str(x_test.shape[0])
    for value in result:
        print "Name: " + str(value['name'])
        print "Accuracy: " + str(value['accuracy'])
        print "Deviation: " + str(value['error'])
        print ""
    err = 0
    for i in range(0, 3):
        if res[i]['name'] != result[i]['name']:
            err += 1
    print err
    print "Accuracy: " + str(1 - (float(err) / 3))
    print("")



