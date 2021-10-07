import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from experiment_wrapper import Experiment
from data_prepper import Cyrillic
import pickle
from sklearn.metrics import classification_report
import time

data_wrappers = [Cyrillic()]
# Validation Curves on rbf
# for d in data_wrappers:
#
#     experiment = Experiment(d, SVC(), test_size=.25)
#     param_range = ['rbf']
#     param_name = "kernel"
#     experiment.plot_hyperparam_learning( param_name, param_range, title="Kernel params", njobs=20, save_img=False)

# -----------Validation chart for C parameter
# for d in data_wrappers:
#     param_range = np.linspace(.01, 10, 5)
#     param_name = "C"
#     experiment = Experiment(d, SVC(), test_size=.25)
#     time_start = time.time()
#     experiment.validation_curve(param_range, param_name, log=False, save_img=False)
#     end_time = time.time() - time_start
#     print('Plot hyperparam time: ' + experiment.classifier.__class__.__name__ + " "
#                                         + param_name + " hyperparam\n" + d.data_name + " " + str(end_time))

# ------------Validation chart for gamma parameter
# for d in data_wrappers:
#     param_range = np.linspace(20, 40, 5)
#     param_name = "gamma"
#     experiment = Experiment(d, SVC(verbose=False), test_size=.25)
#     experiment.data_wrapper.data.to_csv("pca_cyrillic.csv", index=False)
#     time_start = time.time()
#     experiment.validation_curve(param_range, param_name, log=False, save_img=True)
#     end_time = time.time() - time_start
#     print('Plot hyperparam time: ' + experiment.classifier.__class__.__name__ + " "
#                                         + param_name + " hyperparam\n" + d.data_name + " " + str(end_time))

# ----- Optimal parameter run for SVM Classifier
optimal_params = [
{'C': 1.9, 'gamma': 16, 'kernel': 'rbf'},
]
for i in range(0, len(data_wrappers)):
    data_wrapper = data_wrappers[i]
    param_kv = optimal_params[i]

    experiment = Experiment(data_wrapper, SVC(**param_kv), test_size=.20)
    # experiment.plot_learning_curve("Learning Curve for " + experiment.classifier.__class__.__name__+ " " +
    #                                data_wrapper.data_name + " optimal params", save_img=True)

    clf = SVC(**param_kv, random_state=42)
    cv_scores = cross_val_score(clf, experiment.X_train, experiment.y_train, cv=5, n_jobs=-1)
    print("Average 5-Fold CV Score for " + experiment.classifier.__class__.__name__ + " ("
          + data_wrapper.data_name + ") : {}".format(np.mean(cv_scores)))

    clf = SVC(C=1.9, gamma=16, kernel='rbf')
    clf.fit(experiment.X_train, experiment.y_train)
    y_pred = clf.predict(experiment.X_test)
    print(classification_report(y_true=experiment.y_test, y_pred=y_pred))

    filename = 'SVC.sav'
    pickle.dump(clf, open(filename, 'wb'))
