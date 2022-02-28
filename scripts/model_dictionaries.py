from random import uniform
from scipy.stats import randint, uniform
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier




model_dictionary = {"EL":{"model": LogisticRegression(solver="saga", penalty="elasticnet"),
                          "param_search":{"l1_ratio": uniform(0, 1),
                                          "max_iter": randint(100, 2000),
                                          "C": uniform(0, 1)}},

                    "LR":{"model": LogisticRegression(solver="saga", penalty="none"),
                          "param_search":{"max_iter": randint(100, 2000),
                                          "C": uniform(0, 1)}},

                    "RF":{"model": RandomForestClassifier(),
                          "param_search":{"n_estimators": randint(1, 100),
                                          "criterion":["gini", "entropy"],
                                          "max_depth": randint(1, 10),
                                          "max_samples": uniform(0, 1),
                                          "bootstrap":[True, False]}},

                    "GB":{"model":GradientBoostingClassifier(),
                          "param_search":{"learning_rate": uniform(0.0001, 1-0.0001),
                                         "n_estimators":randint(1, 100),
                                         "subsample": uniform(0.1, 0.5),
                                         "max_depth": randint(1, 10)}},

                    "LLA":{"model":LogisticRegression(solver="saga", penalty="l1"),
                          "param_search":{"max_iter": randint(100, 2000),
                                          "C": uniform(0, 0.5)}},

                    "DT":{"model": DecisionTreeClassifier(),
                          "param_search":{"criterion":["gini", "entropy"],
                                          "splitter":["best", "random"],
                                          "max_depth": randint(1, 10)}},
                    
                    "SGD":{"model": SGDClassifier(penalty="elasticnet"),
                          "param_search":{"loss":["log", "modified_huber"],
                                          "alpha": uniform(0.0001, 1),
                                          "l1_ratio": uniform(0, 1),
                                          "epsilon": uniform(0, 1)}},
                    "LD": {"model": LinearDiscriminantAnalysis(tol=1e-4),
                           "param_search":{"solver":["eigen", "lsqr"],
                                           "shrinkage": uniform(0, 1)}},

                    "HGB":{"model": HistGradientBoostingClassifier(),
                           "param_search": {"learning_rate": uniform(0, 1),
                           "max_iter": randint(10, 200),
                           "l2_regularization": uniform(0,1),
                           "max_depth": randint(1, 10)}}}