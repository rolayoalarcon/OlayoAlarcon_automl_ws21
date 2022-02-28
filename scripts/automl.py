import itertools
import pandas as pd
from tqdm import tqdm
from scripts.hyperband import hyperband
from scripts.evaluation import evaluate_models
from scripts.model_dictionaries import model_dictionary
from scripts.preprocessing_functions import split_dataset, data_preprocessing, basic_processing
from scripts.preprocessing_dictionaries import category_strategy_dict, numerical_strategy_dict
from sklearn.pipeline import Pipeline


def automl(X_complete, y_complete, test_fraction=0.2, num_iterations=5, fidelity_parameter=2, loss_function="neg_log_loss", cv_folds=10, random_state=42, categorical_strategies = ["OHE", "BDE", "SUM"], numerical_strategies=["SSE", "MMS", "LOG"], classification_algorithms=["EL", "RF", "DT"], n_jobs=1, n_estimators="full"):
       
       # Separate into test and training sets
       separated_datasets = split_dataset(X_complete, y_complete, test_fraction, random_state)
           
       # Remove NAs and add a small pseudocount
       separated_datasets["X_train_clean"] = basic_processing(separated_datasets["X_train"])
       separated_datasets["X_validation_clean"] = basic_processing(separated_datasets["X_validation"])
       
       # Get data with categorical features
       encoded_data, encoder_objs = data_preprocessing(separated_datasets["X_train_clean"], categorical_strategies, numerical_strategies, category_strategy_dict, numerical_strategy_dict)
       
       # Perform hyperband
       complete_archive = pd.DataFrame()
       best_dict = {}

       for prep_strat, model in tqdm(itertools.product(encoded_data.keys(), classification_algorithms), desc="Data-Model combinations", position=0):
              combination_id = f"{prep_strat}-{model}"
              #print(combination_id)
              archive_df, best_obj = hyperband(encoded_data[prep_strat], separated_datasets["y_train"], num_iterations, fidelity_parameter, loss_function, model, random_state, cv_folds, n_jobs, n_estimators)

              # Add to the Archive
              archive_df["combination"] = combination_id
              complete_archive = pd.concat([complete_archive, archive_df])

              # Prepare best dictionary
              best_dict[combination_id] = {"estimator": best_obj, "preprocessor": encoder_objs[prep_strat]}
       
       # Final evaluation
       best_model, best_model_tag, best_evaluation = evaluate_models(best_dict, separated_datasets["X_validation_clean"], separated_datasets["y_validation"], loss_function)

       # Final preparation
       X_clean = basic_processing(X_complete)

       _, _, classifier = best_model_tag.split("-")
       
       final_encoder = best_model["preprocessor"]
       final_classifier = model_dictionary[classifier]["model"]
       final_classifier.set_params(**best_model["estimator"].best_params_)

       ## Pipeline
       final_pipeline = Pipeline([('scaler', final_encoder), ('classifier', final_classifier)])
       final_pipeline.fit(X_clean, y_complete)

       ## For analysis
       complete_archive["champion"] = "not"
       complete_archive.loc[complete_archive["combination"] == best_model_tag, "champion"] = "champion"

       return complete_archive, final_pipeline, best_evaluation