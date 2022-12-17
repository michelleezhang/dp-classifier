# pip install aif360
# pip install aif360[LawSchoolGPA]
# pip install aif360[Reductions]
# pip install lime
# pip install BlackBoxAuditing

# Importing necessary libraries and dependencies
import sys
sys.path.insert(1, "../")  

import numpy as np
np.random.seed(0)

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing # importing pre-processing DIR algo
from aif360.algorithms.preprocessing import DisparateImpactRemover # importing pre-processing DIR algo
from aif360.datasets import CompasDataset # importing our dataset

from IPython.display import Markdown, display

import sys
sys.path.insert(0, '../')

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# LIME
from aif360.datasets.lime_encoder import LimeEncoder
import lime
from lime.lime_tabular import LimeTabularExplainer


from collections import defaultdict
import pandas as pd

np.random.seed(1)



# Compute bias of ProPublica dataset using the disparate_impact metric

from aif360.datasets import CompasDataset

dataset_orig = CompasDataset(
    protected_attribute_names=['race'],
    privileged_classes=[lambda x: x == 'Caucasian'],
    features_to_drop=['sex', 'age']
    )

dataset_train, dataset_test = dataset_orig.split([0.7], shuffle=True)

privileged_groups = [{'race': 1}] 
unprivileged_groups = [{'race': 0}]

dataset_train1 = BinaryLabelDatasetMetric(dataset_train, 
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
print(dataset_train1.disparate_impact())

# Implement a logistic regression classifier to predict if a defendant is likely to re-offend
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from collections import defaultdict

dataset_copy = dataset_orig.copy()
x, _ = dataset_copy.convert_to_dataframe()
y, _ = dataset_copy.convert_to_dataframe()

x.drop('two_year_recid', inplace=True, axis=1)
y = y.loc[:, 'two_year_recid']

(dataset_orig_train, dataset_orig_val, dataset_orig_test) = dataset_copy.split([0.5, 0.8], shuffle=True)

sens_ind = 0
sens_attr = dataset_orig_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in dataset_orig_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in dataset_orig_train.privileged_protected_attributes[sens_ind]]

dataset = dataset_orig_train
model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
fit_params = {'logisticregression__sample_weight': dataset.instance_weights}

lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

y_val_pred_prob = model.predict_proba(dataset.features)
pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]

y_val_pred = (y_val_pred_prob[:, pos_ind] > 0.5).astype(np.float64)

# TODO #3
# Compute the fairness of the classification algorithm according to the
# disparate_impact metric and the equal_opportunity_difference metrics

dataset_pred = dataset.copy()
dataset_pred.labels = y_val_pred
metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

print(metric.disparate_impact())
print(metric.equal_opportunity_difference())

# Remove disparate impact on the original training data
# Compute the disparate_impact metric once more.

from aif360.algorithms.preprocessing import DisparateImpactRemover 

DIR = DisparateImpactRemover(repair_level=1.0, sensitive_attribute='race')

dataset_DIR_train = DIR.fit_transform(dataset_orig_train)

metric_DIR_train = BinaryLabelDatasetMetric(dataset_DIR_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
print(metric_DIR_train.disparate_impact())


# TODO #5
# Implement bias mitigation by reweighing the original training data
# Compute the disparate_impact metric once more

RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

dataset_RW_train = RW.fit_transform(dataset_orig_train)

metric_RW_train = BinaryLabelDatasetMetric(dataset_RW_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)

print(metric_RW_train.disparate_impact())

# Using the same logistic regression classifier as above on the transformed data
# from 7b, compute the fairness of the classifier using the classification
# disparate_impact and equal_opportunity_difference metrics

dataset_copy = dataset_DIR_train.copy()
x, _ = dataset_copy.convert_to_dataframe()
y, _ = dataset_copy.convert_to_dataframe()

x.drop('two_year_recid', inplace=True, axis=1)
y = y.loc[:, 'two_year_recid']

(dataset_orig_train, dataset_orig_val, dataset_orig_test) = dataset_copy.split([0.5, 0.8], shuffle=True)

sens_ind = 0
sens_attr = dataset_orig_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in dataset_orig_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in dataset_orig_train.privileged_protected_attributes[sens_ind]]

dataset = dataset_orig_train
model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
fit_params = {'logisticregression__sample_weight': dataset.instance_weights}

lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

y_val_pred_prob = model.predict_proba(dataset.features)
pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]

y_val_pred = (y_val_pred_prob[:, pos_ind] > 0.5).astype(np.float64)

dataset_pred = dataset.copy()
dataset_pred.labels = y_val_pred
metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

print(metric.disparate_impact())
print(metric.equal_opportunity_difference())