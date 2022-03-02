import pandas as pd
from sklearn.metrics import SCORERS

def evaluate_models(model_dict, X_test, y_test, scorer):
    """
    Evaluate models on a held out test set
    Parameters
    ----------
    model_dict: pandas dataframe 
        contains the features used for prediction for all samples
    
    X_test: pandas dataframe, numpy array
        contains the features to be used in prediction

    y_test: numpy array
        labels for prediction

    Returns
    -------
    dict, str, dataframe
       1. A dictionary with the best preprocessor-classifier combo
       2. Tag of the best model
       3. Dataframe with final evaluation results
    """

    complete_evaluation = pd.DataFrame()
    # Iterate over best possible candidates
    for mod_combo in model_dict.keys():
        # Gather estimator and preprocessor
        estimator = model_dict[mod_combo]["estimator"].best_estimator_
        transformer = model_dict[mod_combo]["preprocessor"]

        # Transform data
        X_transformed = transformer.transform(X_test)

        # Evaluation
        evaluation = SCORERS[scorer](estimator, X_transformed, y_test)

        # Add result to dataframe
        eval_df = pd.DataFrame({"evaluation": [evaluation], "model": [mod_combo]}, index=[mod_combo])
        complete_evaluation = pd.concat([complete_evaluation, eval_df])

    # Select best model
    best_overall = complete_evaluation.loc[complete_evaluation["evaluation"] == complete_evaluation["evaluation"].max(), "model"].to_list()[0]

    return model_dict[best_overall], best_overall, complete_evaluation


