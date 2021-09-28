import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve

df = pd.read_csv("scaled_cyrillic.csv", index_col="Unnamed: 0")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = KNeighborsClassifier()

param_range = range(1, 200, 10)
train_scores, test_scores = validation_curve(
    clf, X, y, param_name="n_neighbors", param_range=param_range,
    scoring="accuracy", n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with " + clf.__class__.__name__)
plt.xlabel("param")
plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange",)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", )
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy",)
plt.legend(loc="best")
plt.show()

# value = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
# print(value)
