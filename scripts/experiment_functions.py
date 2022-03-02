import pandas as pd
from scripts.automl import automl
from sklearn.metrics import SCORERS
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from scripts.preprocessing_functions import basic_processing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier


def kfold_experiments(X_complete, y_complete, cv_folds, scorer, automl_params, voter_classifiers, n_cores=1, random_state=42, use_processor=False):
    i =0 
    skf = StratifiedKFold(n_splits=cv_folds)

    automl_archive = pd.DataFrame()
    kfold_archive = pd.DataFrame()
    automl_test = pd.DataFrame()

    for train_index, test_index in skf.split(X_complete, y_complete):
        print(f"Fold {i}")

        # Gather the data
        X_train = basic_processing(X_complete.iloc[train_index,:])
        y_train = y_complete[train_index]
    
        X_test = basic_processing(X_complete.iloc[test_index, :])
        y_test = y_complete[test_index]

        # Training
        ## AutoML
        automl_params["X_complete"] = X_train
        automl_params["y_complete"] = y_train

        complete_archive, best_pipeline, final_test = automl(**automl_params)

        # Processing?
        if use_processor:
            best_preprocessor = best_pipeline.named_steps["scaler"]
            X_train = best_preprocessor.transform(X_train)
            X_test = best_preprocessor.transform(X_test)


        ## Random Forrest
        rf = RandomForestClassifier(random_state=random_state, n_jobs=n_cores)
        rf.fit(X_train, y_train)

        ## Voter
        mv = VotingClassifier(estimators=[("clf1", voter_classifiers[0]), ("clf2", voter_classifiers[1]), ("clf", voter_classifiers[2])],
                              voting='soft', n_jobs=n_cores)
        mv.fit(X_train, y_train)

        # Testing
        mv_score = SCORERS[scorer](mv, X_test, y_test)
        rf_score = SCORERS[scorer](rf, X_test, y_test)
        am_score = SCORERS[scorer](best_pipeline, basic_processing(X_complete.iloc[test_index, :]), y_test)
        
        k_performance = pd.DataFrame({"model": ["AutoML", "RF", "MV"],
                                    f"{scorer}": [mv_score, rf_score, am_score],
                                    "kfold":[i, i, i]})
        
        # Add Performance to dataframe
        kfold_archive = pd.concat([kfold_archive, k_performance])
        
        # Keep info about kfold for automl
        complete_archive["kfold"] = i
        automl_archive = pd.concat([automl_archive, complete_archive])
        automl_test = pd.concat([automl_test, final_test])
        
        i += 1
    return kfold_archive, automl_archive, automl_test


