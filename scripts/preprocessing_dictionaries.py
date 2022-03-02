import numpy as np
import category_encoders as ce
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler

def ohe_famd(x):
    x_proba = np.sqrt((x.sum(axis=0) / x.shape[0]) + 1e-4)
    x_div = x / x_proba
    x_center = x_div - x_div.mean(axis=0)
    
    return x_center

category_strategy_dict = {"OHE": OneHotEncoder(sparse=False, handle_unknown='ignore'),
                          "BDE": ce.BackwardDifferenceEncoder(),
                          "SUM": ce.SumEncoder(),
                          "OHE_VAR":Pipeline([("categorical_selection", OneHotEncoder(sparse=False, handle_unknown='ignore')),
                                             ("threshold", VarianceThreshold(0.8*(1-0.8)))]),
                          "OHE_SPR": Pipeline([("categorical_selection", OneHotEncoder(sparse=False, handle_unknown='ignore')),
                                             ("transform", FunctionTransformer(ohe_famd))])}

numerical_strategy_dict = {"SSE": StandardScaler(),
                           "MMS": MinMaxScaler(clip=True),
                           "LOG": FunctionTransformer(np.log2),
                           "SSE_VAR": Pipeline([("scaling", StandardScaler()),
                                                ("dim_reduction", VarianceThreshold(0.7))]),
                            "LOG_VAR":Pipeline([("scaling", FunctionTransformer(np.log2)),
                                                ("dim_reduction", VarianceThreshold(0.7))])}