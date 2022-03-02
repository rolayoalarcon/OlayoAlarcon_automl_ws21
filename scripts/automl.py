import itertools
import pandas as pd
from tqdm import tqdm
from scripts.hyperband import optimise_parameters
from scripts.evaluation import evaluate_models
from scripts.model_dictionaries import model_dictionary
from scripts.preprocessing_functions import split_dataset, data_preprocessing, basic_processing, data_reduction
from scripts.preprocessing_dictionaries import category_strategy_dict, numerical_strategy_dict
from sklearn.pipeline import Pipeline


def automl(X_complete, y_complete, test_fraction=0.2, n_searches=5, fidelity_parameter=2, score_function="neg_log_loss", categorical_strategies = ["OHE", "BDE", "SUM"], numerical_strategies=["SSE", "MMS", "LOG"], dim_reduction = None, n_dims = None, classification_algorithms=["EL", "RF", "DT"], cv_folds=10, n_estimators="full", optimizer_strategy="my_hyperband", random_state=42, n_jobs=1):
       """
       Performs the complete automl pipline
       Parameters
       ----------
       X_complete: pandas dataframe 
              contains the features used for prediction for all samples.
       
       y_complete: np array 
              must be 1D. Contains the labels to predict for each sample
       
       test_fraction: float
              the fraction of total samples that will be reserved for final testing
       
       n_searches: int
              the number of hyperband brackets to explore. Each bracket will start with more samples than the last
       
       fidelity_parameter: int
              the halving parameter, which determines the proportion of samples that are added for each subsequent iteration.
       
       score_function: str 
              the score function to MAXIMIZE. Must be one of the options from sklearn.metrics.SCORERS
       
       categorical_strategies: list
              a list of preprocessing strategies for categorical variables.
              Possible options are
              "OHE": One Hot Encoding
              "SUM": Sum Contrast Encoding
              "BDE": Backward Difference Encoder
              "OHE_VAR": One Hot Encoding followed by Variance Thresholding
              "OHE_SPR": One Hot Encoding followed by division of sqrt of probability

       numerical_strategies: list
             a list of preprocessing strategies for numerical variables. 
             Possible options are
             "SSE": Standard Scaler
             "MMS": Min Max Scaler
             "LOG: Log transformation
             "SSE_VAR": Standard Scaler followed by Variance Thresholding
             "LOG_VAR": Log transformation followed by Variance Thresholding
       
       dim_reduction: list, None
              a list of dimension reduction techniques. At the moment only "FAMD" is available. Default is None
       
       n_dims: int, None
              number of dimensions that the dataset should be reduced to.
       
       classification_algorithms: list
              a list of classifiers to try out during the procedure.
              If optimizer_strategy is hyperband or bohb only RF, DT, GB, EL are available.
              Options include
              "EL": Elastic Net
              "RF": Random Forrest
              "DT": Decision Tree
              "LR": Logistic Regression
              "GB": Gradient Boosting Classifier
              "HGB": Histogram Gradient Booster
              "SGD": Stochastic Gradient Descent
              "LLA": LASSO
              "LD": Linear Discriminant Analysis
       
       cv_folds: int
              number of stratefied cross validations to carry out in training
       
       n_estimators: int, "full"
              number of initial candidates at each halfsearch space. If "full" then the number of inital candidates is determined automatically.
       
       optimizer_strategy: str,
              Can be
              "my_hyperband": my implementation of hyperband
              "hyperband": hyperbland implementation of hpbandster-sklearn (https://hpbandster-sklearn.readthedocs.io/en/latest/#hpbandster-sklearn)
              "bohb": BOHB implementation of hpbandster-sklearn
       
       random_state: int
              seed for reproducibility
       
       n_jobs: int
              number of resources for parallelization
       
       Returns
       -------
       dataframe, Pipeline, dataframe
              1. complete information from Hyperband search. 
              2. The best preprocess-classifier pipeline object found
              3. final evaluations for model selection
    """
       
       # Separate into test and training sets
       separated_datasets = split_dataset(X_complete, y_complete, test_fraction, random_state)
           
       # Remove NAs and add a small pseudocount
       separated_datasets["X_train_clean"] = basic_processing(separated_datasets["X_train"])
       separated_datasets["X_validation_clean"] = basic_processing(separated_datasets["X_validation"])
       
       # Get data with categorical features
       encoded_data, encoder_objs = data_preprocessing(separated_datasets["X_train_clean"], categorical_strategies, numerical_strategies, category_strategy_dict, numerical_strategy_dict)

       # Apply dimensionality reduction if necessary
       if dim_reduction != None:
              encoded_data, encoder_objs = data_reduction(separated_datasets["X_train_clean"], encoded_data, encoder_objs, dim_reduction, n_dims)
       
       # Perform hyperband
       complete_archive = pd.DataFrame()
       best_dict = {}

       # Perform hyperband for all preprocess - classifier combinations
       for prep_strat, model in tqdm(itertools.product(encoded_data.keys(), classification_algorithms), desc="Data-Model combinations", position=0):
              combination_id = f"{prep_strat}-{model}"

              # Perform search
              archive_df, best_obj = optimise_parameters(encoded_data[prep_strat], separated_datasets["y_train"], n_searches, fidelity_parameter, score_function, model, random_state, cv_folds, n_jobs, n_estimators, optimizer_strategy)

              # Add to the Archive
              archive_df["combination"] = combination_id
              complete_archive = pd.concat([complete_archive, archive_df])

              # Prepare best dictionary
              best_dict[combination_id] = {"estimator": best_obj, "preprocessor": encoder_objs[prep_strat]}
       
       # Final evaluation
       best_model, best_model_tag, best_evaluation = evaluate_models(best_dict, separated_datasets["X_validation_clean"], separated_datasets["y_validation"], score_function)

       # Final preparation
       X_clean = basic_processing(X_complete)

       classifier = best_model_tag.split("-")[-1]
       
       final_encoder = best_model["preprocessor"]
       final_classifier = model_dictionary[classifier]["model"]
       final_classifier.set_params(**best_model["estimator"].best_params_)

       ## Pipeline and final fit
       final_pipeline = Pipeline([('scaler', final_encoder), ('classifier', final_classifier)])
       final_pipeline.fit(X_clean, y_complete)

       ## For analysis
       complete_archive["overall_champion"] = "not"
       complete_archive.loc[(complete_archive["HS_champion"] == "HS_champion") & 
                            (complete_archive["Hyp_champion"]=="Hyp_champion") &
                            (complete_archive["combination"] == best_model_tag), "overall_champion"] = "overall_champion"

       return complete_archive, final_pipeline, best_evaluation