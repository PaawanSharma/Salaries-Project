[![Code style**: black](https**://img.shields.io/badge/code%20style-black-000000.svg)](https**://github.com/psf/black)# Salaries prediction project in Python## The problemTwo tables of jobs are given, with each row corresponding to a different joband columns corresponding to various features of those jobs. For the trainingset, salaries are also given for each job. The task is to construct a modelthat can predict the salaries for the test set.## Directory structure\* **README.md**: this report.\* **code/Salaries prediciton.ipynb**: the main script of code, solving the problem from start to finish.\* **code/**: also contains other code scripts used by the main notebook.    \* **code/eda/**: scripts used for exploratory data analysis.         \* **code/eda/plot.py**: functions used for creating plots in EDA.         \* **code/eda/stats.py**: code used for statistical analysis in EDA.    \* **code/feature_engineering/encoders.py**: code used for encoding categorical variables.    \* **code/exceptions.py**: contains an exception used by encoding functions.    \* **code/model_selection.py**: code used for cross-validation of algorithms.    \* **code/preprocessing.py**: code used for preparing datasets for machine learning.\* **data/**: the data used in this problem.    \* **data/train_features.csv**: the jobs of the training set.    \* **data/train_salaries.csv**: the true salaries for the training set jobs.    \* **data/test_features.csv**: the jobs of the test set.\* **cross_val_logs/**: logs recording cross-validations that were performed.    \* **cross_val_logs/train_data_cross_val.csv**: cross-validation log for the training data.    \* **cross_val_logs/engineered_train_data_cross_val.csv**: cross-validation log for the training data after feature engineering had been performed on it.\* **results/**: the best model found. This directory will contain unqiuesubdirectories for each model deployed.    \* **results/_model\_subdirectory_/model.joblib**: the saved model which can be loaded into a Python script using the joblib module.    \* **results/_model\_subdirectory_/feature_importances.csv**: feature importances for the saved model.    \* **results/_model\_subdirectory_/predictions.csv**: the model's predictions for the test set salaries.\* **.gitignore**: files to be ignored by git.