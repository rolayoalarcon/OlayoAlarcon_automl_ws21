from random import uniform
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier




model_dictionary = {"EL":{"model": LogisticRegression(solver="saga", penalty="elasticnet"),
                          "param_search":{"l1_ratio": uniform(0, 1),
                                          "max_iter": randint(100, 2000),
                                          "C": uniform(0, 1)}},

                    "RF":{"model": RandomForestClassifier(),
                          "param_search":{"n_estimators": randint(1, 100),
                                          "criterion":["gini", "entropy"],
                                          "max_depth": randint(1, 10),
                                          "max_features": randint(1, 6),
                                          "bootstrap":[True, False]}},

                    "GB":{"model":GradientBoostingClassifier(),
                          "param_search":{"learning_rate": uniform(0.0001, 1-0.0001),
                                         "n_estimators":randint(1, 100),
                                         "subsample": uniform(0.1, 0.5),
                                         "max_depth": randint(1, 10)}}
                                         }