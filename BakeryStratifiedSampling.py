import pandas as pd
import RegressorRecommender as rr
import csv
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split

rec = rr.RegressorRecommender()

df = pd.read_csv("bakery_clean.csv")

rec.set_data(df.copy())
rec.define_target('Qty')
rec.run()

fn = 'result/b_stratified_result.csv'
os.remove(fn) if os.path.exists(fn) else None

with open(fn, 'wb') as csvfile:
    logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    logwriter.writerow(['Data: House Price'])
    logwriter.writerow(['Type: Regression'])
    logwriter.writerow(['Data Count: ' + str(df.shape[0])])
    logwriter.writerow(['Sampling Type: Random'])
    logwriter.writerow(['---'])
    logwriter.writerow(['All Data Result (RMSE): '])
print "All Data Result: "
res_acc = rec.sort('rmse')
co = 1
for value in res_acc:
    with open(fn, 'ab') as csvfile:
        logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow([str(co), value['name']])
        co += 1
    print value['name']
    print value['accuracy']
    print value['time']
    print ''

res_time = rec.sort('time')
co = 1

with open(fn, 'ab') as csvfile:
    logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    logwriter.writerow(['All Data Result (Time): '])
print "All Data Result(Time): "
for value in res_acc:
    with open(fn, 'ab') as csvfile:
        logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow([str(co), value['name']])
        co += 1
    print value['name']
    print value['accuracy']
    print value['time']
    print ''

print("---")


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


with open(fn, 'ab') as csvfile:
    logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    logwriter.writerow(['---'])
    logwriter.writerow(['Percentage Sampling', 'Acc. Regression Tree', 'Acc. SVR', 'Acc. Lasso Regression', 'Time. Regression Tree', 'Time. SVR', 'Time. Lasso Regression', 'Acc. Recommender', 'Acc. Runtime Recommender', 'Total Time'])
sample_df = df.copy()
x = df.copy()
y = df['Qty']
for case in range(1, 6):
    with open(fn, 'ab') as csvfile:
        logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow(['Iteration-' + str(case)])
    print "Iterarion -" + str(case)
    for p in frange(0.1, 0.6, 0.1):
        bins = np.linspace(0, y.shape[0], 5)
        y_binned = np.digitize(y, bins)
        x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, stratify=y_binned)

        rec.reset()
        rec.set_data(x_test.copy())
        rec.define_target('Qty')

        start = time.clock()
        rec.run()
        end = time.clock()
        total_time = end - start
        result = rec.sort('rmse')
        with open(fn, 'ab') as csvfile:
            c = {}
            print "Percent: " + str(p) + ", Rows: " + str(x_test.shape[0])
            for value in result:
                print "Name: " + str(value['name'])
                print "Accuracy: " + str(value['accuracy'])
                print "Deviation: " + str(value['error'])
                print ""
                c[value['name']] = {
                    'accuracy': value['accuracy'],
                    'deviation': value['error'],
                    'time': value['time']
                }
            err = 0
            for i in range(0, 3):
                if res_acc[i]['name'] != result[i]['name']:
                    err += 1
            result = rec.sort('time')
            err_time = 0
            for i in range(0, 3):
                if res_time[i]['name'] != result[i]['name']:
                    err_time += 1
            print "Accuracy: " + str(1 - (float(err_time) / 3))
            print("")
            logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            logwriter.writerow([str(p), c['Regression Tree']['accuracy'], c['SVR']['accuracy'], c['Lasso Regression']['accuracy'], c['Regression Tree']['time'], c['SVR']['time'], c['Lasso Regression']['time'], (1 - (float(err) / 3)), (1 - (float(err_time) / 3)), total_time])