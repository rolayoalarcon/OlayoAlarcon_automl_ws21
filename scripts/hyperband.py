from tqdm import tqdm
import numpy as np
import pandas as pd
from scripts.model_dictionaries import model_dictionary
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.utils.validation import check_is_fitted
from hpbandster_sklearn import HpBandSterSearchCV


def my_hyperband(X_train, y_train, num_tracks, eta, loss_function, model, random_state, cv_folds, n_jobs, n_estimators):
    """
    My implementation of hyperband based on HalvingRandomSearch

    Parameters
    ----------
    X_train: pandas dataframe 
        contains the features used for prediction for all samples.
       
    y_train: np array 
        must be 1D. Contains the labels to predict for each sample
       
    num_tracks: int
        number of HalvingRandomSearch iterations to run

    eta: int
        the halving parameter, which determines the proportion of samples that are added for each subsequent iteration.
       
   loss_function: str 
        the score function to MAXIMIZE. Must be one of the options from sklearn.metrics.SCORERS
       
    model: str
        the model to be used in the search
    
    random_state: int
        seed for reproducibility
    
    cv_folds: int
        number of stratefied cross validations to carry out in training
    
    n_jobs: int
        number of resources for parallelization

    n_estimators: int, "full"
        number of initial candidates at each halfsearch space. If "full" then the number of inital candidates is determined automatically.
       
    Returns
    -------
        dataframe, estimator
            1. contains information about the halving search iterations.
            2. best model found by the hyperband
    """
    # Estimate the inital number of samples given the hyperband iterations and the total number of samples
    min_initial_budget = np.ceil(np.exp(-((num_tracks * np.log(eta)) - np.log(X_train.shape[0])))).astype(int)

    hyper_best_dict = {}
    archive_df = pd.DataFrame()
    comparison_df = pd.DataFrame()

    # Iterate over hyperband iterations
    for i in tqdm(range(num_tracks), desc="Hyperband iterations", position=1):
        
        # Minimal budget given the iteration
        min_budget = min_initial_budget * (eta**i)

        # Get the inital object
        model_obj = model_dictionary[model]["model"]

        # Set some parameters for seed and ncores
        if model in ["LR", "EL"]:
            model_obj.set_params(n_jobs=n_jobs)
        elif model in ["GB", "DT", "HGB"]:
            model_obj.set_params(random_state=random_state)
        elif model in ["RF", "SGD"]:
            model_obj.set_params(n_jobs=n_jobs, random_state=random_state)

        # Gather the parameter generator
        param_dist = model_dictionary[model]["param_search"]

        # Prepare Halving search
        if n_estimators == "full":
            HalvingSearch = HalvingRandomSearchCV(estimator=model_obj, 
                                                param_distributions=param_dist, 
                                                factor=eta, 
                                                random_state=random_state, 
                                                min_resources=min_budget,
                                                cv=cv_folds,
                                                scoring=loss_function,
                                                n_jobs=n_jobs,
                                                refit=True)
        else:
            HalvingSearch = HalvingRandomSearchCV(estimator=model_obj, 
                                                param_distributions=param_dist, 
                                                factor=eta, 
                                                random_state=random_state, 
                                                min_resources=min_budget,
                                                cv=cv_folds,
                                                scoring=loss_function,
                                                n_jobs=n_jobs,
                                                n_candidates=n_estimators,
                                                refit=True)

        # Dot the search
        HalvingSearch.fit(X_train, y_train)
        
        # Add the iteration to the archive
        result_df = pd.DataFrame(HalvingSearch.cv_results_)
        result_df["hyperband_iter"] = i
        result_df["HS_champion"] = "not"
        winning_params = HalvingSearch.cv_results_["params"][HalvingSearch.best_index_]
        result_df.loc[result_df["params"].astype(str) == str(winning_params), "HS_champion"] = "HS_champion"

        archive_df = pd.concat([archive_df, result_df])

        # Add the estimator to the dictionary
        hyper_best_dict[i] = HalvingSearch
        performance_df = pd.DataFrame({"Hyp_iteration": [i],
                                      "performance":[HalvingSearch.best_score_]})
        comparison_df = pd.concat([comparison_df, performance_df])
    
    # Once hyperband is finished select the best model
    best_model = comparison_df.loc[comparison_df["performance"] == comparison_df["performance"].max(),
                                   "Hyp_iteration"].tolist()[0]
    best_estimator = hyper_best_dict[best_model]

    archive_df["Hyp_champion"] = "not"
    archive_df.loc[(archive_df["HS_champion"]=="HS_champion") & 
                   (archive_df["hyperband_iter"] == best_model),
    "Hyp_champion"] = "Hyp_champion"


    return archive_df, best_estimator

def hpband_search(X_train, y_train, num_tracks, eta, loss_function, random_state, cv_folds, n_jobs, strategy, model):
    """
    The hyperband implementation of hpbandster-sklearn

    Parameters
    ----------
    X_train: pandas dataframe 
        contains the features used for prediction for all samples.
       
    y_train: np array 
        must be 1D. Contains the labels to predict for each sample
       
    num_tracks: int
        number of HalvingRandomSearch iterations to run

    eta: int
        the halving parameter, which determines the proportion of samples that are added for each subsequent iteration.
       
   loss_function: str 
        the score function to MAXIMIZE. Must be one of the options from sklearn.metrics.SCORERS
    
    random_state: int
        seed for reproducibility
    
    cv_folds: int
        number of stratefied cross validations to carry out in training
    
    n_jobs: int
        number of resources for parallelization
    
    strategy: str,
        wither "hyperband" or "bohb"

    model: str
        the model to be used in the search
       
    Returns
    -------
        dataframe, estimator
            1. contains information about the halving search iterations.
            2. best model found by the hyperband
    """
    # Prepare model
    model_obj = model_dictionary[model]["model"]
    model_param = model_dictionary[model]["param_config"]

    # Carry out search
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
    """
    The hyperband implementation of hpbandster-sklearn

    Parameters
    ----------
    X_train: pandas dataframe 
        contains the features used for prediction for all samples.
       
    y_train: np array 
        must be 1D. Contains the labels to predict for each sample
       
    num_tracks: int
        number of HalvingRandomSearch iterations to run

    eta: int
        the halving parameter, which determines the proportion of samples that are added for each subsequent iteration.
       
   loss_function: str 
        the score function to MAXIMIZE. Must be one of the options from sklearn.metrics.SCORERS
    
    model: str
        the model to be used in the search

    random_state: int
        seed for reproducibility
    
    cv_folds: int
        number of stratefied cross validations to carry out in training
    
    n_jobs: int
        number of resources for parallelization
    
    n_estimators: int, "full"
        number of initial candidates at each halfsearch space. If "full" then the number of inital candidates is determined automatically.
    
    opt_strategy: str,
        wither "hyperband", "bohb" or "my_hyperband
       
    Returns
    -------
        dataframe, estimator
            1. contains information about the halving search iterations.
            2. best model found by the optimizer
    """
    
    # Decide which function to use
    if opt_strategy == "my_hyperband":
        archive, best_model = my_hyperband(X_train, y_train, num_tracks, eta, loss_function, model, random_state, cv_folds, n_jobs, n_estimators)
    else:
        archive, best_model = hpband_search(X_train, y_train, num_tracks, eta, loss_function, random_state, cv_folds, n_jobs, opt_strategy, model)
    
    return archive, best_model
