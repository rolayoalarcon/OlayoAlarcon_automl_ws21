import numpy as np
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler

category_strategy_dict = {"OHE": OneHotEncoder(sparse=False, handle_unknown='ignore'),
                          "BDE": ce.BackwardDifferenceEncoder(),
                          "SUM": ce.SumEncoder()}

numerical_strategy_dict = {"SSE": StandardScaler(),
                           "MMS": MinMaxScaler(clip=True),
                           "LOG": FunctionTransformer(np.log2)}


