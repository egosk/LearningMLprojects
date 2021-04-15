# k nearest neighbors algorithm to classify cars
# data set from https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
# learning_ML with https://www.techwithtim.net/tutorials/machine-learning-python/k-nearest-neighbors-1/


import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')

# converting data to numeric
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# build and train model
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

# show how it works on unique elements of test data
for x in range(len(predicted)):
    print('Predicted: ', names[predicted[x]], ' Data: ', x_test[x], 'Actual: ', names[y_test[x]])

    # see the neighbours of given data point
    # n = model.kneighbors([x_test[x]], 9, True)
    # print('N: ', n)