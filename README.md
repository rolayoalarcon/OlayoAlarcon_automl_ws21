# OlayoAlarcon_automl_ws21
This is my project for the Automatic Machine Learning course taken at Ludwig-Maximilians-Universitaet Muenchen during the Winter Semester 21/22.  

In this main directory you will find the following.

+ **experiments.ipynb**: Contains the 10 fold cross validation comparison of my automl pipeline to a Random Forrest and a Majority Voter. Figures and brief discusssion are presented in a nice report.
+ **experiments.py**: The code is the same as in **experiments.ipynb**, it simply makes it easier to run on a server.
+ **automl_compliance,pdf**: Contains a signed declaration of compliance with examination rules.
+ **requirements.txt**: Lists all required packages and versions used in this project
+ **data**: Directory containing the input for the experiments. 
+ **output**: Directory containing the output of **experiments.ipynb** (and therefore also experiments.py). This includes data tables and figures.
+ **scripts**: This is where the bread and butter of the automl pipeline is. Every script here contributes to the pipeline. The entrypoint function for the pipeline is found in scripts/automl.py

While the project is coded in python, I definitely use a more functional programming approach. Therefore, the entrypoint into the pipeline is the a function called *automl()*. The pipeline is based on the Hyperband algorithm, and relies on the [HalvingRandomSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html#sklearn.model_selection.HalvingRandomSearchCV) function from scikit-learn. I later added a wrapper to HpBandster via the [hpbandster-sklearn](https://hpbandster-sklearn.readthedocs.io/en/latest/) package.   
  
Documentation for the input parameters of my *automl()* function can be found in [automl/automl](automl/automl). However, for convenience, I will paste the documentation below. Hope this works!  
  
Input parameters for the AutoML pipeline!

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
              "hyperband": hyperband implementation of hpbandster-sklearn (https://hpbandster-sklearn.readthedocs.io/en/latest/#hpbandster-sklearn)
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
