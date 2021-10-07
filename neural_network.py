import pandas as pd
from sklearn.model_selection import train_test_split, \
    cross_val_score, GridSearchCV, validation_curve, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import numpy as np
from experiment_wrapper import Experiment
from data_prepper import Cyrillic
import time

data_wrappers = [Cyrillic()]

# layer_sizes = [x for x in itertools.product(range(1, 10), repeat=3)]

# ------ CAUTION: Gridsearch takes a long time!
# optimal_params = [
# {'hidden_layer_sizes': layer_sizes},
# {'hidden_layer_sizes': layer_sizes},
# ]
# counter = 0
# for d in data_wrappers:
#     experiment = Experiment(d, MLPClassifier(random_state=42))
#     print(d.data_name)
#     param_dict = optimal_params[counter]
#     experiment.search_optimal_params("RandomGridSearchCV", param_dict)
#
#     counter += 1

# --------Validation chart for layer_size
# layer_sizes = []
# value = 25
# for i in range(0,50):
#     if len(layer_sizes) <= 0:
#         layer_sizes.append((value,))
#     else:
#         last_val = layer_sizes[-1]
#         last_val = last_val + (value,)
#         layer_sizes.append(last_val)
#
# layer_sizes = layer_sizes
# for d in data_wrappers:
#     param_range = layer_sizes
#     param_name = "hidden_layer_sizes"
#     experiment = Experiment(d, MLPClassifier(), test_size=.25)
#     time_start = time.time()
#     experiment.validation_curve(param_range, param_name, save_img=True, hidden_layer_size='layers')
#     end_time = time.time() - time_start
#     print('Plot hyperparam time: ' + experiment.classifier.__class__.__name__ + " "
#                                         + param_name + " hyperparam\n" + d.data_name + " " + str(end_time))
#
# # --------Validation chart for node amounts per layer
layer_sizes = [(500,)
               ]
for d in data_wrappers:
    param_range = layer_sizes
    param_name = "hidden_layer_sizes"
    experiment = Experiment(d, MLPClassifier(), test_size=.25)
    time_start = time.time()
    experiment.validation_curve(param_range, param_name, save_img=False, hidden_layer_size='nodes')
    end_time = time.time() - time_start
    print('Plot hyperparam time: ' + experiment.classifier.__class__.__name__ + " "
          + param_name + " hyperparam\n" + d.data_name + " " + str(end_time))

# --------Validation chart for activation functions
# for d in data_wrappers:
#     param_range = ['tanh', 'relu', 'identity']
#     param_name = "activation"
#     experiment = Experiment(d, MLPClassifier(hidden_layer_sizes=(25,)), test_size=.25)
#     time_start = time.time()
#     experiment.plot_hyperparam_learning(param_name, param_range,
#                                         "Learning Curve for " + experiment.classifier.__class__.__name__ + " "
#                                         + param_name + " hyperparam\n" + d.data_name, save_img=False)
#     end_time = time.time() - time_start
#     print('Plot hyperparam time: ' + experiment.classifier.__class__.__name__ + " "
#                                         + param_name + " hyperparam\n" + d.data_name + " " + str(end_time))

# ---------Validation chart for max_iter parameter
# for d in data_wrappers:
#     param_range = range(1, 1000, 20)
#     param_name = "max_iter"
#     experiment = Experiment(d, MLPClassifier(), test_size=.25)
#     time_start = time.time()
#     experiment.validation_curve(param_range, param_name, log=False, save_img=True)
#     end_time = time.time() - time_start
#     print('Plot hyperparam time: ' + experiment.classifier.__class__.__name__ + " "
#                                         + param_name + " hyperparam\n" + d.data_name + " " + str(end_time))

# ------- Optimal Params experiment
# optimal_params = [
# {'hidden_layer_sizes': (47, 2, 22, 7, 33), 'activation': 'relu'},
# {'hidden_layer_sizes': (2, 5, 8), 'activation': 'tanh'},
# ]
# for i in range(0, len(data_wrappers)):
#     data_wrapper = data_wrappers[i]
#     param_kv = optimal_params[i]
#
#     experiment = Experiment(data_wrapper, MLPClassifier(**param_kv), test_size=.25)
#     experiment.plot_learning_curve("Learning Curve for " + experiment.classifier.__class__.__name__+ " " +
#                                    data_wrapper.data_name + " optimal params", save_img=False)
#
#     clf = MLPClassifier(**param_kv, random_state=42)
#
#     time_start = time.time()
#     cv_scores = cross_val_score(clf, experiment.X, experiment.y, cv=5, n_jobs=-1)
#     print("Average 5-Fold CV Score for " + experiment.classifier.__class__.__name__ + " ("
#           + data_wrapper.data_name + ") : {}".format(np.mean(cv_scores)))
#
#     runtime = time.time() - time_start
#     print('Run time: ' + str(runtime))
# #
# #     plt.xlabel("iteration")
# #     plt.ylabel("loss")
# #     plt.plot(range(0, len(clf.loss_curve_), clf.loss_curve_))
