# support vector machines for cancer classification
# on dataset from sklearn
# learning ML with https://www.techwithtim.net/tutorials/machine-learning-python/svm-1/
# kernel is function used to represent data in higher dimension f(x1, x2) -> x3

import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print('Label: ', cancer.target_names)

x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


clf = svm.SVC(kernel='linear', C=2) # C -how many points can be on wrong side of kernel - soft margin, default =1
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test) # predict values for test data
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)