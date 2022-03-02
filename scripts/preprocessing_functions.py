import itertools
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scripts.preprocessing_dictionaries import category_strategy_dict, numerical_strategy_dict

def data_reduction(X_original, encoded_data, encoder_objs, dim_reduction, n_dims):

    updated_data = encoded_data.copy()
    updated_objs = encoder_objs.copy()

    if "FAMD" in dim_reduction:
        _, ct_obj = data_preprocessing(X_original, ["OHE_SPR"], ["SSE"], category_strategy_dict, numerical_strategy_dict)
        complete_pipeline = Pipeline([("col_preprocess", ct_obj["OHE_SPR-SSE"]),
                                     ("PCA", PCA(n_components=n_dims))])

        famd_data = complete_pipeline.fit_transform(X_original)
        updated_data["FAMD"] = famd_data
        updated_objs["FAMD"] = complete_pipeline
    
    return updated_data, updated_objs


def basic_processing(feature_df, pseudocount=1e-4):
    """
    Removing NAs from data. Also adding a small pseudocount
    Parameters
    ----------
    feature_df: pandas dataframe 
        contains the features used for prediction for all samples
    
    pseudocount: float
        a small value added to numerical features so as to avoid trouble with log transform.
        Default is 1e-4

    Returns
    -------
    pandas dataframe
        a dataframe that has no more NAs
    """
    original_index = feature_df.index
    categoric_df = feature_df.select_dtypes(include=['object']).copy()
    numerical_df = feature_df.select_dtypes(include=['float64', "int64"]).copy()

    # Replace categoric NAs with 'missing'
    categoric_df.fillna("missing", inplace=True)

    # Replace numeric NAs with the mean of their column
    SI = SimpleImputer(missing_values=np.nan, strategy="mean")
    numerical_noNA = pd.DataFrame(SI.fit_transform(numerical_df), columns=numerical_df.columns)

    ### Add a small pseudocount to the resulting values to avoid trouble with log transformation
    numerical_noNA = numerical_noNA + pseudocount

    # Join the two dframes. We assume row order is maintained
    categoric_df.index = original_index
    numerical_noNA.index = original_index
    feature_clean = numerical_noNA.join(categoric_df)

    return feature_clean




def split_dataset(X_complete, y_complete, test_fraction, random_state):
    """
    Splitting the data into Test and Train splits
    Parameters
    ----------
    X_complete: pandas dataframe 
        contains the features used for prediction for all samples
    
    y_compelte: pandas dataframe 
        contains the class of all samples

    test_fraction: float, int
        if int, should refer to the number of samples that will be used for testing. If a float < 0, it represents the 
        fraction of the complete dataset that will be used for testing

    Returns
    -------
    dict
        a dictionary with the dataset split
    """

    X_train, X_validation, y_train, y_validation = train_test_split(X_complete, y_complete, 
                                                                    test_size=test_fraction, 
                                                                    stratify=y_complete, 
                                                                    random_state=random_state)
    
    dataset_dict = {"X_train": X_train,
                    "X_validation": X_validation,
                    "y_train": y_train,
                    "y_validation": y_validation}

    return(dataset_dict)


def data_preprocessing(features_df, cat_strategies, num_strategies, cat_dict, num_dict):
    """
    Pre-processing categorical and numerical data
    Parameters
    ----------
    feature_df: pandas dataframe 
        contains the features used for prediction for all samples
    
    cat_strategies: list
        a list that contains the codes for the pre-processing strategies to be used for categorical features.
        Elements must match the keys of cat_dict

    num_strategies: list
        a list that contains the codes for the pre-processing strategies to be used for numeric features.
        Elements must match the keys of num_dict
    
    cat_dict: dict
        a dictionary that contains the objects that perform categorical transformation.

    num_dict: dict
        a dictionary that contains the objects that perform numeric transformation.

    Returns
    -------
    dict, dict
        a dictionary with the transformed datasets
        a dictionary with the column transformer objects
    """

    categoric_columns = features_df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = features_df.select_dtypes(include=['float64', "int64"]).columns.tolist()

    dataset_dict = {}
    transformer_dict = {}
    for cat, num in itertools.product(cat_strategies, num_strategies):
        ct = ColumnTransformer([(cat, cat_dict[cat], categoric_columns),
                                (num, num_dict[num], numerical_columns)])
        
        dataset_dict[f'{cat}-{num}'] = ct.fit_transform(features_df)
        transformer_dict[f'{cat}-{num}'] = ct

    return dataset_dict, transformer_dict

    


