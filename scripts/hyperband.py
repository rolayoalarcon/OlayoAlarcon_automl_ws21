from tqdm import tqdm
import numpy as np
import pandas as pd
from scripts.model_dictionaries import model_dictionary
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.utils.validation import check_is_fitted
from hpbandster_sklearn import HpBandSterSearchCV


def my_hyperband(X_train, y_train, num_tracks, eta, loss_function, model, random_state, cv_folds, n_jobs, n_estimators):
    
    min_initial_budget = np.ceil(np.exp(-((num_tracks * np.log(eta)) - np.log(X_train.shape[0])))).astype(int)

    hyper_best_dict = {}
    archive_df = pd.DataFrame()
    comparison_df = pd.DataFrame()

    for i in tqdm(range(num_tracks), desc="Hyperband iterations", position=1):
        min_budget = min_initial_budget * (eta**i)

        model_obj = model_dictionary[model]["model"]

        if model in ["LR", "EL"]:
            model_obj.set_params(n_jobs=n_jobs)
        elif model in ["GB", "DT", "HGB"]:
            model_obj.set_params(random_state=random_state)
        elif model in ["RF", "SGD"]:
            model_obj.set_params(n_jobs=n_jobs, random_state=random_state)

        param_dist = model_dictionary[model]["param_search"]

        if n_estimators == "full":

            HalvingSearch = HalvingRandomSearchCV(estimator=model_obj, 
                                                param_distributions=param_dist, 
                                                factor=eta, 
                                                random_state=random_state, 
                                                min_resources=min_budget,
                                                cv=cv_folds,
                                                scoring=loss_function,
                                                n_jobs=n_jobs)
        else:
            HalvingSearch = HalvingRandomSearchCV(estimator=model_obj, 
                                                param_distributions=param_dist, 
                                                factor=eta, 
                                                random_state=random_state, 
                                                min_resources=min_budget,
                                                cv=cv_folds,
                                                scoring=loss_function,
                                                n_jobs=n_jobs,
                                                n_candidates=n_estimators)


        HalvingSearch.fit(X_train, y_train)
        
        # Add the iteration to the archive
        result_df = pd.DataFrame(HalvingSearch.cv_results_)
        result_df["hyperband_iter"] = i

        archive_df = pd.concat([archive_df, result_df])

        # Add the estimator to the dictionary
        hyper_best_dict[f"Hyp{i}"] = HalvingSearch
        performance_df = pd.DataFrame({"Hyp_iteration": [f"Hyp{i}"],
                                      "performance":[HalvingSearch.best_score_]})
        comparison_df = pd.concat([comparison_df, performance_df])
    
    # Once hyperband is finished select the best model
    best_model = comparison_df.loc[comparison_df["performance"]== comparison_df["performance"].max(),
                                   "Hyp_iteration"].tolist()[0]
    best_estimator = hyper_best_dict[best_model]

    return archive_df, best_estimator

def hpband_search(X_train, y_train, num_tracks, eta, loss_function, random_state, cv_folds, n_jobs, strategy, model):

    model_obj = model_dictionary[model]["model"]
    model_param = model_dictionary[model]["param_config"]

    search = HpBandSterSearchCV(model_obj, 
                            model_param,
                            random_state=random_state, 
                            n_jobs=n_jobs, 
                            n_iter=num_tracks, 
                            verbose=0,
                            resource_name = "n_samples",
                            cv=cv_folds,
                            eta=eta,
                            scoring=loss_function,
                            refit=True, 
                            optimizer=strategy).fit(X_train, y_train)

    return pd.DataFrame(search.cv_results_), search


def optimise_parameters(X_train, y_train, num_tracks, eta, loss_function, model, random_state, cv_folds, n_jobs, n_estimators, opt_strategy):

    if opt_strategy == "my_hyperband":
        archive, best_model = my_hyperband(X_train, y_train, num_tracks, eta, loss_function, model, random_state, cv_folds, n_jobs, n_estimators)
    else:
        archive, best_model = hpband_search(X_train, y_train, num_tracks, eta, loss_function, random_state, cv_folds, n_jobs, opt_strategy, model)
    
    return archive, best_model
