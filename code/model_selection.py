"""
Model Selection.

This module contains two classes:

    * Log - log objects store cross-validation results and interact with
    logfiles.
    * Test_Combination - objects that set up and perform cross-validation and
    send results to log objects.
"""

# Third-party imports
from os.path import isfile
from pandas import DataFrame, read_csv, Series
from sklearn.model_selection import cross_val_score
from time import time as timer

# Local imports
import preprocessing


class Log:
    """Class for logging object.

    Creates a log in memory for storing 5-fold cross-validation results and can
    update logs saved in file. The log can also be retrieved and viewed via its
    dataframe attribute.

    Instance variables
    ------------------
    source -- the existing log file to retrieve at instantiation, if any.

    Public methods
    --------------
    add -- add a cross-validation entry to the log.
    update_logfile -- synchronise with a log saved on file.
    """

    def __init__(self, source=None):
        # If instantiating from file, retrieve the data from the csv file
        if source and isfile(source):
            self.dataframe = read_csv(source, index_col=0)
        # Otherwise, create a blank table for the current log
        else:
            columns = (
                [
                    "Training sample size",
                    "Encoder",
                    "Interactions",
                    "Scaling",
                    "Regressor",
                ]
                + ["CVS_{}_MSE".format(fold) for fold in range(1, 6)]
                + ["Time /s"]
            )
            self.dataframe = DataFrame(columns=columns)

    def add(
        self,
        sample_size,
        encoder,
        interactions,
        scaling,
        regressor,
        cvs,
        train_time,
    ):
        """
        Add an entry to the log.

        Keyword Arguments
        -----------------
        sample_size -- the number of rows trained on.
        encoder -- the encoder object used, if any.
        interactions -- the interaction object used, if any.
        scaling -- the scaler used, if any.
        regressor -- the algorithm and parameters cross-validated.
        cvs -- the cross-validation scores as a list.
        train_time -- the time taken to run cross-validation.
        """
        # Create the entry
        row_data = (
            [sample_size, encoder, interactions, scaling, regressor]
            + cvs
            + [train_time]
        )
        row = Series(row_data, self.dataframe.columns)
        # Add the entry to the log dataframe
        self.dataframe = self.dataframe.append(row, ignore_index=True)

    def update_logfile(self, logpath):
        """Update a log saved in file as a csv or create a new one.

        Keyword Arguments
        -----------------
        logpath -- the path of the csv file containing the saved log.
        """
        # Create a log file if it doesn't exist
        if not isfile(logpath):
            self.dataframe.to_csv(logpath)
        # Update a log file
        else:
            # Add the current log to the bottom of the table
            self.dataframe.to_csv(logpath, mode="a", header=False)
            # Check for and remove duplicate entries
            with_duplicates = read_csv(logpath, index_col=0)
            without_duplicates = with_duplicates.drop_duplicates(
                ignore_index=True
            )
            self.dataframe = without_duplicates
            self.dataframe.to_csv(logpath)
        print("Results saved in log!")


class Test_Combination:
    """Class for cross-validation object.

    Can set up and perform 5-fold cross-validation and store the results in a
    log.

    Instance variables
    ------------------
    encoder -- the encoder to be used, if any.
    interactions -- the interaction object to be used, if any.
    scale -- the scaler to be used, if any.
    regressor -- the regressor to be cross-validated.

    Public methods
    --------------
    run -- perform 5-fold cross-validation.
    log_results -- send the results to a Log object.
    print_summary -- print a summary of cross-validation.
    """

    def __init__(
        self, encoder=None, interactions=None, scale=None, regressor=None
    ):
        self.encoder = encoder
        self.interactions = interactions
        if self.encoder:
            self.encoder_name = self.encoder.__name__
        else:
            self.encoder_name = "NONE"
        self.scale = scale
        self.regressor = regressor
        self.been_run = False

    def run(self, train_data, target, log):
        """
        Perform 5-fold cross validaton and send the results to log.

        Keyword Arguments
        -----------------
        train_data -- the DataFrame containing the training data.
        target -- the name of the target variable column.
        log -- the log object for storing results.
        """
        # Record the number of rows to be trained on
        self.sample_size = train_data.shape[0]
        # Prepare the dataset as features and target and encode it if necessary
        X_train, y_train = preprocessing.encode_and_split(
            train=train_data, target=target, encoder=self.encoder
        )
        # Apply any interactions object passed
        if self.interactions:
            X_train = self.interactions.fit_transform(X_train)
            X_train = DataFrame(X_train)
        # Apply any scaling passed
        if self.scale:
            X_train = self.scale.fit_transform(X_train)

        # Perform cross-validation and time it
        start_time = timer()
        self.cvs = _cv_mse(self.regressor, X_train, y_train)
        end_time = timer()
        self.time = end_time - start_time

        # Record the cross-validation has been performed
        self.been_run = True

        # Send the results to the log and print a summary of the
        # cross-validation
        self.log_results(log)
        self.print_summary()

    def log_results(self, log):
        """
        Send the results of cross-validation to the log object.

        Keyword Arguments
        -----------------
        log -- the log object for storing results.
        """
        log.add(
            self.sample_size,
            self.encoder_name,
            str(self.interactions),
            str(self.scale),
            str(self.regressor),
            self.cvs.tolist(),
            self.time,
        )
        print("Results staged for logging!")

    def print_summary(self):
        """Print a summary of the cross-validation that was performed."""
        print(
            "Mean MSE for {} with {}: {}".format(
                str(self.regressor), self.encoder_name, (self.cvs * -1).mean(),
            )
        )


def _cv_mse(estimator, X, y):
    return cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=True,
    )
