import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from scripts.experiment_functions import kfold_experiments
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def determine_dtype(x):
    return x.decode("utf-8")

if __name__ == "__main__":
    
    # Read the data
    print("Reading")
    churn_data = arff.loadarff("data/chrun.arff")
    churn_df = pd.DataFrame(churn_data[0])
    
    okcupid_stem = arff.loadarff("data/okcupid_stem.arff")
    okcupid_df = pd.DataFrame(okcupid_stem[0])
    
    # Some inital cleaning-up
    print("Cleaining up")
    churn_df["class"] = churn_df["class"].astype(int)
    churn_df["number_customer_service_calls"] = churn_df["number_customer_service_calls"].astype(int)
    
    okcupid_df["job"] = okcupid_df["job"].apply(determine_dtype)
    okcupid_df.replace(b'?', np.nan, inplace=True)
    okcupid_df["income"] = okcupid_df["income"].astype('float64')
    
    
    # Experiments
    ## Churn
    print("Churn experiment")
    # Data
    X_churn = churn_df.drop(columns="class")
    y_churn = np.reshape(churn_df["class"].values, X_churn.shape[0])

    ### Experiment parameters
    kfolds = 10
    scorer = "neg_log_loss"
    ncores = 50
    random_state = 42

    ### AutoML params
    automl_dict = {"classification_algorithms": ["EL", "RF", "GB"],
                   "numerical_strategies": ["SSE", "MMS"],
                   "categorical_strategies": ["OHE", "SUM"],
                   "test_fraction": 0.1,
                   "cv_folds": 4,
                   "random_state": random_state,
                   "num_iterations": 4,
                   "fidelity_parameter": 2,
                   "n_jobs": ncores,
                   "loss_function": scorer}

    ### Voters
    voter_clfs = [RandomForestClassifier(random_state=random_state), 
                  DecisionTreeClassifier(random_state=random_state),
                  GradientBoostingClassifier(random_state=random_state)]
    
    churn_experiment, churn_automl = kfold_experiments(X_churn, y_churn, kfolds, scorer, automl_dict, voter_clfs,
                                                  n_cores=ncores, random_state=42)
    
    ### Write output
    #### Figure
    plt.figure(figsize=(10,10))
    sns.set_theme(style="whitegrid")
    bp = sns.boxplot(x="validation", y="model", data=churn_experiment)
    bp = sns.swarmplot(x="validation", y="model", data=churn_experiment, color=".25")
    bp.get_figure().savefig("data/churn_results/base_boxplot.pdf", bbox_inches='tight')
    
    #### Files
    churn_experiment.to_csv("data/churn_results/churn_experiments.tsv.gz", sep='\t')
    churn_automl.to_csv("data/churn_results/churn_automl.tsv.gz", sep='\t')
    
    ## OkCupid
    # Data
    X_cupid = okcupid_df.drop(columns="job")
    y_cupid = np.reshape(okcupid_df["job"].values, X_cupid.shape[0])
    
    # Experiment parameters
    kfolds = 10
    scorer = "neg_log_loss"
    ncores = 80
    random_state = 42

    # AutoML params
    automl_dict = {"classification_algorithms": ["DT", "RF", "HGB"],
                   "numerical_strategies": ["SSE", "MMS"],
                   "categorical_strategies": ["OHE", "SUM"],
                   "test_fraction": 0.2,
                   "cv_folds": 4,
                   "random_state": random_state,
                   "num_iterations": 4,
                   "fidelity_parameter": 4,
                   "n_jobs": ncores,
                   "loss_function": scorer,
                   "n_estimators": 10}

    # Voters
    voter_clfs = [RandomForestClassifier(random_state=random_state), 
                  DecisionTreeClassifier(random_state=random_state),
                  GradientBoostingClassifier(random_state=random_state)]
    
    print("Cupid Experiment")
    cupid_experiment, cupid_automl = kfold_experiments(X_cupid, y_cupid, kfolds, scorer, automl_dict, voter_clfs,
                                                  n_cores=ncores, random_state=42)
    
    #### Figure
    plt.figure(figsize=(10,10))
    sns.set_theme(style="whitegrid")
    bp = sns.boxplot(x="validation", y="model", data=cupid_experiment)
    bp = sns.swarmplot(x="validation", y="model", data=cupid_experiment, color=".25")
    bp.get_figure().savefig("data/okcupid_results/base_boxplot.pdf", bbox_inches='tight')
    
    #### Files
    cupid_experiment.to_csv("data/okcupid_results/cupid_experiments.tsv.gz", sep='\t')
    cupid_automl.to_csv("data/okcupid_results/cupid_automl.tsv.gz", sep='\t')
    
    
    ## Joint Analysis
    
    churn_performance = churn_automl.loc[(churn_automl["champion"] == "champion") & (churn_automl["hyperband_iter"] == 0), ["iter", "mean_test_score"]]
    
    cupid_performance = cupid_automl.loc[(churn_automl["champion"] == "champion") & (churn_automl["hyperband_iter"] == 0), ["iter", "mean_test_score"]]
    
    combined_performance = pd.concat([churn_performance, cupid_performance])
    
    mean_iter = combined_performance.groupby("iter")["mean_test_score"].mean()
    std_iter = combined_performance.groupby("iter")["mean_test_score"].std()
    x = mean_iter.index
    
    # Compute upper and lower bounds using chosen uncertainty measure: here
    # it is a fraction of the standard deviation of measurements at each
    # time point based on the unbiased sample variance
    lower = mean_iter - std_iter
    upper = mean_iter + std_iter

    # Draw plot with error band and extra formatting to match seaborn style
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(x, mean_iter, label='signal mean')
    ax.plot(x, lower, color='tab:blue', alpha=0.1)
    ax.plot(x, upper, color='tab:blue', alpha=0.1)
    ax.fill_between(x, lower, upper, alpha=0.2)
    ax.set_xlabel('iter')
    ax.set_ylabel('validation_error')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.get_figure().savefig("data/iteration_performance.pdf", bbox_inches="tight")