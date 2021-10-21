from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve, learning_curve
import numpy as np
from sklearn.base import ClassifierMixin
from data_prepper import DataWrapper
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


class Experiment:

    def __init__(self, data_wrapper: DataWrapper, classifier: ClassifierMixin, test_size=0.25, seed=42):
        self.data_wrapper = data_wrapper
        self.data_wrapper.load_data()
        self.test_size = test_size
        X, y = self.data_wrapper.get_train_test()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        self.X = X
        self.y = y
        self.classifier = classifier
        self.seed = seed

    def search_optimal_params(self, param_search_cv, param_dict, reset_classifier=True,
                              n_iter=1000, cv=5, verbose=0, n_jobs=-1):
        if param_search_cv == "GridSearchCV":
            grid = GridSearchCV(self.classifier, param_dict, cv=cv, verbose=verbose, n_jobs=n_jobs)
        else:
            grid = RandomizedSearchCV(self.classifier, param_dict, n_iter=n_iter, cv=cv, verbose=verbose, n_jobs=n_jobs)

        grid.fit(self.X_train, self.y_train)
        file=open("gridsearch.txt", 'a+')
        print(self.data_wrapper.data_name, file=file)
        print(self.classifier.__class__.__name__, file=file)
        print(grid.best_estimator_, file=file)
        print(grid.best_params_, file=file)
        print(grid.best_score_, file=file)

        if reset_classifier:
            self.reset_classifier()

    def reset_classifier(self):

        get_class = lambda x: globals()[x]
        c = get_class(self.classifier.__class__.__name__)
        self.classifier = c()

    def validation_curve(self, param_range, param_name, plot_type="plot", log=False, save_img=True, hidden_layer_size=None, ):

        # https://scikit-learn.org/stable/auto_examples/model_selection/
        # plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py

        train_scores, test_scores = validation_curve(
            self.classifier, self.X_train, self.y_train, param_name=param_name,
            param_range=param_range,
            cv=5, scoring="accuracy", n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if hidden_layer_size == "layers":
            param_range = [len(i) for i in param_range]
        elif hidden_layer_size == "nodes":
            param_range = [i[0] for i in param_range]

        if plot_type == "bar":
            plt.title("Validation Scores " + self.data_wrapper.data_name + " " + self.classifier.estimator_type)
            plt.xlabel(param_name)
            plt.ylabel("Accuracy Score")
            plt.bar(param_range, train_scores_mean, label="Training score",
                    color="darkorange")
            # plt.fill_between(param_range, train_scores_mean - train_scores_std,
            #                  train_scores_mean + train_scores_std, alpha=0.2,
            #                  color="darkorange")
            plt.bar(param_range, test_scores_mean, label="Cross-validation score",
                    color="navy")
            # plt.fill_between(param_range, test_scores_mean - test_scores_std,
            #                  test_scores_mean + test_scores_std, alpha=0.2,
            #                  color="navy")
            plt.legend(loc="best")
            if save_img:
                suffix = "_" + hidden_layer_size if not None else ""
                plt.savefig("images/validation_plot_" + self.data_wrapper.data_name + "_" +
                            self.classifier.__class__.__name__ + "_" + param_name + suffix + ".png")
            plt.show()
            plt.clf()

        elif plot_type == "scatter":
            plt.title("Validation Scores " + self.data_wrapper.data_name + " " + self.classifier.estimator_type)
            plt.xlabel(param_name)
            plt.ylabel("Accuracy Score")
            plt.scatter([str(val) for val in param_range], train_scores_mean, label="Training",
                    color="darkorange")
            # plt.fill_between(param_range, train_scores_mean - train_scores_std,
            #                  train_scores_mean + train_scores_std, alpha=0.2,
            #                  color="darkorange")
            plt.scatter([str(val) for val in param_range], test_scores_mean, label="Cross-validation",
                    color="navy")
            # plt.fill_between(param_range, test_scores_mean - test_scores_std,
            #                  test_scores_mean + test_scores_std, alpha=0.2,
            #                  color="navy")
            plt.legend(loc="best")
            if save_img:
                plt.savefig("images/validation_plot_" + self.data_wrapper.data_name + "_" +
                            self.classifier.__class__.__name__ + "_" + param_name + ".png")
            plt.show()
            plt.clf()
        else:
            plt.title("Validation Curve " + self.data_wrapper.data_name + " (Test size: " +str(self.test_size) + ")\n"
                      + self.classifier.__class__.__name__)
            suffix = "_" + hidden_layer_size if hidden_layer_size is not None else ""
            plt.xlabel(param_name + suffix)
            plt.ylabel("Accuracy Score")
            if not log:
                plt.plot(param_range, train_scores_mean, label="Training score",
                         color="darkorange")
            else:
                plt.plot(param_range, train_scores_mean, label="Training score",
                         color="darkorange")
            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2,
                             color="darkorange")
            if not log:
                plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                         color="navy")
            else:
                plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                         color="navy")
            plt.fill_between(param_range, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2,
                             color="navy")
            plt.legend(loc="best")
            if save_img:
                plt.savefig("images/validation_plot_" + self.data_wrapper.data_name + "_" +
                            self.classifier.__class__.__name__ + "_" + param_name + suffix + ".png")
            plt.show()
            plt.clf()


    def plot_learning_curve(self, title, cv=5, njobs=-1, save_img=True):

        # Validation curves courtesy scikit website:
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy Score")
        plt.grid()

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(self.classifier, self.X_train, self.y_train, cv=cv, n_jobs=njobs,
                           train_sizes=np.linspace(.1, 1.0, 200),
                           return_times=True, scoring="accuracy")
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="orange")
        plt.plot(train_sizes, train_scores_mean, #'o-',
                     label="Training score", color="b")
        plt.plot(train_sizes, test_scores_mean, #'D-',
                     label="Cross-validation score", color="orange")
        plt.legend(loc="best")
        if save_img:
            img_string = title.replace(" ", "_").replace("\n", "_").lower()
            plt.savefig("images/"+img_string)
        plt.show()
        plt.clf()

        # plot fit time learning curve
        time_title = "Fit times " + self.classifier.__class__.__name__ + " " + self.data_wrapper.data_name
        plt.title(time_title)
        plt.xlabel("Training examples")
        plt.ylabel("Fit time")
        plt.grid()

        plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1,
                             color="b")
        plt.plot(train_sizes, fit_times_mean, #'D-',
                     label="Fit times", color="b")
        plt.legend(loc="best")
        if save_img:
            img_string = time_title.replace(" ", "_").replace("\n", "_").lower()
            plt.savefig("images/"+img_string)
        plt.show()
        plt.clf()
        self.reset_classifier()

    def plot_hyperparam_learning(self, param_name, param_range, title, cv=5,
                                 test_size=0.25, ylim=None, njobs=-1, save_img=True):

        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()
        for hyper_val in param_range:

            self.classifier.__setattr__(param_name, hyper_val)
            train_sizes, train_scores, test_scores, fit_times, _ = \
                learning_curve(self.classifier, self.X_train, self.y_train, cv=cv, n_jobs=njobs,
                               train_sizes=np.linspace(.1, 1.0, 10),
                               return_times=True, scoring="accuracy", random_state=self.seed)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            fit_times_mean = np.mean(fit_times, axis=1)
            fit_times_std = np.std(fit_times, axis=1)

            plt.plot(train_sizes, train_scores_mean, 'o-',
                     label=str(hyper_val) + " Training score")
            color1 = plt.gca().lines[-1].get_color()
            plt.plot(train_sizes, test_scores_mean, 'D-',
                     label=str(hyper_val) + " Cross-validation score", color=color1)
            # color2 = plt.gca().lines[-1].get_color()
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color=color1)
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color=color1)
            self.reset_classifier()

        plt.legend(loc="best")
        if save_img:
            img_string = title.replace(" ", "_").replace("\n", "_").lower()
            plt.savefig("images/"+img_string)
        plt.show()
