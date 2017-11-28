import pandas as pd
import ClassifierRecommender as cr

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


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


sample_df = df.copy()
for p in frange(0.1, 0.6, 0.1):
    row = sample_df.shape[0]

    percent = p

    sdf = sample_df.sample(int(row * percent))

    rec.reset()
    rec.set_data(sdf.copy())
    rec.define_target('left')
    rec.run()
    result = rec.sort('accuracy')
    print "Percent: " + str(p) + ", Rows: " + str(sdf.shape[0])
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
