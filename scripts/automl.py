import random
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from preprocessing_functions import split_dataset, data_preprocessing, basic_processing
from preprocessing_dictionaries import category_strategy_dict, numerical_strategy_dict


def automl(X_complete, y_complete, test_fraction, random_state=42,
           categorical_strategies = ["OHE", "BDE", "SUM"], numerical_strategies=["SSE", "MMS", "LOG"]):

    print(numerical_strategies)

    # Separate into test and training sets
    separated_datasets = split_dataset(X_complete, y_complete, test_fraction, random_state)

    # Remove NAs and add a small pseudocount
    separated_datasets["X_train_clean"] = basic_processing(separated_datasets["X_train"])
    separated_datasets["X_validation_clean"] = basic_processing(separated_datasets["X_validation"])



    # Get data with categorical features
    encoded_data, encoder_objs = data_preprocessing(separated_datasets["X_train_clean"], 
                                                    categorical_strategies, numerical_strategies,
                                                    category_strategy_dict, numerical_strategy_dict)

    # Perform hyperband

    return encoded_data








    
