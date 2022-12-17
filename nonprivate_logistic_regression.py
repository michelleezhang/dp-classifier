import diffprivlib.models as dp
import numpy as np
from sklearn.linear_model import LogisticRegression 

from sklearn import datasets
dataset = datasets.load_breast_cancer()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

clf = LogisticRegression(solver="lbfgs")
clf.fit(x_train, y_train)

baseline = clf.score(x_test, y_test)
print("Non-private test accuracy: %.2f%%" % (baseline * 100))