# !pip install ai-privacy-toolkit
# !pip install diffprivlib
# !pip install adversarial-robustness-toolbox

from sklearn import datasets
dataset = datasets.load_breast_cancer()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.5)

from sklearn.linear_model import LogisticRegression
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

model = LogisticRegression(solver="lbfgs", max_iter=3000)
model.fit(x_train, y_train)

art_classifier = ScikitlearnLogisticRegression(model)
base_model_accuracy = model.score(x_test, y_test)

print('Base model accuracy: ', base_model_accuracy)

import numpy as np
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

attack_train_ratio = 0.5
attack_train_size = int(len(x_train) * attack_train_ratio)
attack_test_size = int(len(x_test) * attack_train_ratio)

attack = MembershipInferenceBlackBox(art_classifier, attack_model_type='rf') 

#train attack model
attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
           x_test[:attack_test_size], y_test[:attack_test_size])

# infer attacked feature
inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])

# check accuracy
train_acc = np.sum(inferred_train) / len(inferred_train)
test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
print('attack accuracy on training data: ', train_acc)
print('attack accuracy on test data: ', test_acc)
print('overall attack accuracy: ', acc)


import diffprivlib.models as dp

dp_model = dp.LogisticRegression(epsilon=2, data_norm=2)
dp_model.fit(x_train, y_train)
# print('norm: ', np.linalg.norm(x_train) )

dp_art_model = ScikitlearnLogisticRegression(dp_model)
print('DP model accuracy: ', dp_model.score(x_test, y_test))

dp_attack = MembershipInferenceBlackBox(dp_art_model, attack_model_type='rf')

# train attack model
dp_attack.fit(x_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size],
              x_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size])

# infer 
dp_inferred_train = dp_attack.infer(x_train.astype(np.float32)[attack_train_size:], y_train[attack_train_size:])
dp_inferred_test = dp_attack.infer(x_test.astype(np.float32)[attack_test_size:], y_test[attack_test_size:])

# check accuracy
dp_train_acc = np.sum(dp_inferred_train) / len(dp_inferred_train)
dp_test_acc = 1 - (np.sum(dp_inferred_test) / len(dp_inferred_test))
dp_acc = (dp_train_acc * len(dp_inferred_train) + dp_test_acc * len(dp_inferred_test)) / (len(dp_inferred_train) + len(dp_inferred_test))
print('attack accuracy on training data: ', dp_train_acc)
print('attack accuracy on test data: ', dp_test_acc)
print('overall attack accuracy: ', dp_acc)




accuracy = []
attack_accuracy = []
epsilons = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 75.0, 100.0, 200.0]

for eps in epsilons:
    dp_clf = dp.LogisticRegression(epsilon=eps, data_norm=5)
    dp_clf.fit(x_train, y_train)
    accuracy.append(dp_clf.score(x_test, y_test))
    dp_art_classifier = ScikitlearnLogisticRegression(dp_clf)
    dp_attack = MembershipInferenceBlackBox(dp_art_classifier, attack_model_type='rf')
    dp_attack.fit(x_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size].astype(np.float32),
                  x_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size].astype(np.float32))
    dp_inferred_train = dp_attack.infer(x_train.astype(np.float32)[attack_train_size:], y_train.astype(np.float32)[attack_train_size:])
    dp_inferred_test = dp_attack.infer(x_test.astype(np.float32)[attack_train_size:], y_test.astype(np.float32)[attack_train_size:])
    dp_train_acc = np.sum(dp_inferred_train) / len(dp_inferred_train)
    dp_test_acc = 1 - (np.sum(dp_inferred_test) / len(dp_inferred_test))
    dp_acc = (dp_train_acc * len(dp_inferred_train) + dp_test_acc * len(dp_inferred_test)) / (len(dp_inferred_train) + len(dp_inferred_test))
    attack_accuracy.append(dp_acc)


import matplotlib.pyplot as plt

plt.plot(epsilons, accuracy)
plt.plot(epsilons, np.ones_like(epsilons) * base_model_accuracy, dashes=[2,2], label="base model")
plt.title("Differentially private logistic regression")
plt.xlabel("epsilon")
plt.ylabel("Model accuracy")
plt.ylim(0, 1)
plt.xlim(0.1, 200)
plt.show()

plt.plot(epsilons, attack_accuracy)
plt.plot(epsilons, np.ones_like(epsilons) * acc, dashes=[2,2], label="base model")
plt.title("Differentially private logistic regression")
plt.xlabel("epsilon")
plt.ylabel("Attack accuracy")
plt.ylim(0, 1)
plt.xlim(0.1, 200)
plt.show()