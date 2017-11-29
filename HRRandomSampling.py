import pandas as pd
import ClassifierRecommender as cr
import csv
import numpy as np
import os
import time

rec = cr.ClassifierRecommender()

df = pd.read_csv("hr.csv")
df.drop(['sales', 'salary'], inplace=True, axis=1)

rec.set_data(df.copy())
rec.define_target('left')
rec.run()

fn = 'result/hr_random_result.csv'
os.remove(fn) if os.path.exists(fn) else None

with open('result/hr_random_result.csv', 'wb') as csvfile:
    logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    logwriter.writerow(['Data: HR'])
    logwriter.writerow(['Type: Classification'])
    logwriter.writerow(['Data Count: ' + str(df.shape[0])])
    logwriter.writerow(['Sampling Type: Random'])
    logwriter.writerow(['---'])
    logwriter.writerow(['All Data Result (Acc): '])
print "All Data Result: "
res_acc = rec.sort('accuracy')
co = 1
for value in res_acc:
    with open('result/hr_random_result.csv', 'ab') as csvfile:
        logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow([str(co), value['name']])
        co += 1
    print value['name']

res_time = rec.sort('time')
co = 1

with open('result/hr_random_result.csv', 'ab') as csvfile:
    logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    logwriter.writerow(['All Data Result (Time): '])

for value in res_time:
    with open('result/hr_random_result.csv', 'ab') as csvfile:
        logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow([str(co), value['name']])
        co += 1
    print value['name']
print("---")


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


with open('result/hr_random_result.csv', 'ab') as csvfile:
    logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    logwriter.writerow(['---'])
    logwriter.writerow(['Percentage Sampling', 'Acc. Decision Tree', 'Acc. Naive Bayes', 'Acc. Logistic Regression', 'Time. Decision Tree', 'Time. Naive Bayes', 'Time. Logistic Regression', 'Acc. Recommender', 'Acc. Runtime Recommender', 'Total Time'])
sample_df = df.copy()
for case in range(1, 6):
    with open('result/hr_random_result.csv', 'ab') as csvfile:
        logwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow(['Iteration-' + str(case)])
    for p in frange(0.1, 0.6, 0.1):
        row = sample_df.shape[0]

        percent = p

        sdf = sample_df.sample(int(row * percent))

        rec.reset()
        rec.set_data(sdf.copy())
        rec.define_target('left')
        start = time.clock()
        rec.run()
        end = time.clock()
        total_time = end - start
        result = rec.sort('accuracy')
        with open('result/hr_random_result.csv', 'ab') as csvfile:
            c = {}
            print "Percent: " + str(p) + ", Rows: " + str(sdf.shape[0])
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
            logwriter.writerow([str(p), c['Decision Tree']['accuracy'], c['Naive Bayes']['accuracy'], c['Logistic Regression']['accuracy'], c['Decision Tree']['time'], c['Naive Bayes']['time'], c['Logistic Regression']['time'], (1 - (float(err) / 3)), (1 - (float(err_time) / 3)), total_time])
