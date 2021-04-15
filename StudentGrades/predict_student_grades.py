# linear regression algorithm to predict students final grade based on series of attributes
# data set from https://archive.ics.uci.edu/ml/datasets/Student+Performance
# learning_ML with https://www.techwithtim.net/tutorials/machine-learning-python/linear-regression/

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


data = pd.read_csv('StudentGrades/student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

#dividing data into features (what we use for prediction) and label (what we want to predict)
predict = 'G3'
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# traing model multiple times and save best one
best = 0
for _ in range(20):
    # dividing data into training  and testing
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # define model, train and test it
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print('Acc: ', str(acc))

    # saving best model
    if acc > best:
        best = acc
        with open('StudentGrades/student_grades.pickle', 'wb') as f:
            pickle.dump(linear, f)

# loading model
pickle_in = open('StudentGrades/student_grades.pickle', 'rb')
linear = pickle.load(pickle_in)

# print("-------------------------")
# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)
# print("-------------------------")
#
# predicted= linear.predict(x_test)
# for x in range(len(predicted)):
#     print(predicted[x], x_test[x], y_test[x])

# plotting model
plot = 'G1' # or other features
plt.scatter(data[plot], data['G3'])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel('Final Grade')
plt.show()

