import diffprivlib.models as dp
import numpy as np
from sklearn.linear_model import LogisticRegression 

from sklearn import datasets
dataset = datasets.load_breast_cancer()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)



dp_clf = dp.LogisticRegression(random_state=0)
dp_clf.fit(x_train, y_train)

print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" % 
     (dp_clf.epsilon, dp_clf.score(x_test, y_test) * 100))



accuracy = []
epsilons = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 75.0, 100.0, 200.0]

for eps in epsilons:
    dp_clf = dp.LogisticRegression(epsilon=eps, data_norm=5)
    dp_clf.fit(x_train, y_train)
    accuracy.append(dp_clf.score(x_test, y_test))

import matplotlib.pyplot as plt

plt.plot(epsilons, accuracy)
plt.plot(epsilons, np.ones_like(epsilons) * baseline, dashes=[2,2], label="base model")
plt.title("Differentially private logistic regression")
plt.xlabel("epsilon")
plt.ylabel("Model accuracy")
plt.ylim(0, 1)
plt.xlim(epsilons[0], epsilons[-1])
plt.show()