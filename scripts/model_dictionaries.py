import numpy as np
from random import uniform
from scipy.stats import randint, uniform
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier


# Config space
l1_ratio = CSH.UniformFloatHyperparameter('l1_ratio', lower=0, upper=1, log=False)
max_iter = CSH.UniformIntegerHyperparameter("max_iter", lower=10, upper=1000)
C = CSH.UniformFloatHyperparameter("C", lower=0, upper=1)
criterion_rf =  CSH.CategoricalHyperparameter("criterion", choices=["gini", "entropy"])
max_depth = CSH.UniformIntegerHyperparameter("max_depth", lower=1, upper=10)
max_samples = CSH.UniformFloatHyperparameter("max_samples", lower=0, upper=1)
bstrap = CSH.CategoricalHyperparameter("bootstrap", choices=[True, False])
learning_rate = CSH.UniformFloatHyperparameter("learning_rate", lower=0, upper=1)
n_estimators = CSH.UniformIntegerHyperparameter("n_estimators", lower=1, upper=100)
subsample = CSH.UniformFloatHyperparameter("subsample", lower=0.1, upper=0.5)
splitter = CSH.CategoricalHyperparameter("splitter", choices=["best", "random"])

# Specific configs
rf = CS.ConfigurationSpace(42)
rf.add_hyperparameter(n_estimators)
rf.add_hyperparameter(criterion_rf)
rf.add_hyperparameter(max_depth)
rf.add_hyperparameter(max_samples)
rf.add_hyperparameter(bstrap)

el = CS.ConfigurationSpace(42)
el.add_hyperparameter(C)
el.add_hyperparameter(max_iter)
el.add_hyperparameter(l1_ratio)

gb = CS.ConfigurationSpace(42)
gb.add_hyperparameter(learning_rate)
gb.add_hyperparameter(n_estimators)
gb.add_hyperparameter(subsample)
gb.add_hyperparameter(max_depth)

dt = CS.ConfigurationSpace(42)
dt.add_hyperparameter(criterion_rf)
dt.add_hyperparameter(splitter)
dt.add_hyperparameter(max_depth)

model_dictionary = {"EL":{"model": LogisticRegression(solver="saga", penalty="elasticnet"),
                          "param_search":{"l1_ratio": uniform(0, 1),
                                          "max_iter": np.random.randint(100, 2000),
                                          "C": uniform(0, 1)},
                          "param_config": el},

                    "LR":{"model": LogisticRegression(solver="saga", penalty="none"),
                          "param_search":{"max_iter": randint(100, 2000),
                                          "C": uniform(0, 1)}},

                    "RF":{"model": RandomForestClassifier(),
                          "param_search":{"n_estimators": randint(1, 100),
                                          "criterion":["gini", "entropy"],
                                          "max_depth": randint(1, 10),
                                          "max_samples": uniform(0, 1),
                                          "bootstrap":[True, False]},
                          
                          "param_config": rf},

                    "GB":{"model":GradientBoostingClassifier(),
                          "param_search":{"learning_rate": uniform(0.0001, 1-0.0001),
                                         "n_estimators":randint(1, 100),
                                         "subsample": uniform(0.1, 0.5),
                                         "max_depth": randint(1, 10)},
                          "param_config":gb},

                    "LLA":{"model":LogisticRegression(solver="saga", penalty="l1"),
                          "param_search":{"max_iter": randint(100, 2000),
                                          "C": uniform(0, 0.5)}},

                    "DT":{"model": DecisionTreeClassifier(),
                          "param_search":{"criterion":["gini", "entropy"],
                                          "splitter":["best", "random"],
                                          "max_depth": randint(1, 10)},
                         "param_config":dt},
                    
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



