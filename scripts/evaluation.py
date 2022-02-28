import pandas as pd
from sklearn.metrics import SCORERS

def evaluate_models(model_dict, X_test, y_test, scorer):

    complete_evaluation = pd.DataFrame()
    for mod_combo in model_dict.keys():
        estimator = model_dict[mod_combo]["estimator"].best_estimator_
        transformer = model_dict[mod_combo]["preprocessor"]

        X_transformed = transformer.transform(X_test)

        evaluation = SCORERS[scorer](estimator, X_transformed, y_test)

        eval_df = pd.DataFrame({"evaluation": [evaluation], "model": [mod_combo]}, index=[mod_combo])
        complete_evaluation = pd.concat([complete_evaluation, eval_df])

    best_overall = complete_evaluation.loc[complete_evaluation["evaluation"] == complete_evaluation["evaluation"].max(), "model"].to_list()[0]

    return model_dict[best_overall], best_overall, complete_evaluation


