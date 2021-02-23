"""
"""

# Third-party imports
from os.path import isfile
from pandas import DataFrame, read_csv, Series
from sklearn.model_selection import cross_val_score
from time import time as timer

# Local imports
import preprocessing


class Log:
    def __init__(self, source=None):
        if source and isfile(source):
            self.dataframe = read_csv(source, index_col=0)
        else:
            columns = (
                [
                    "Training sample size",
                    "Encoder",
                    "Interactions",
                    "Scaling",
                    "Regressor",
                ]
                + ["CSV_{}_MSE".format(fold) for fold in range(1, 6)]
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
        row_data = (
            [sample_size, encoder, interactions, scaling, regressor]
            + cvs
            + [train_time]
        )
        row = Series(row_data, self.dataframe.columns)
        self.dataframe = self.dataframe.append(row, ignore_index=True)

    def update_logfile(self, logpath):
        if not isfile(logpath):
            self.dataframe.to_csv(logpath)
        else:
            self.dataframe.to_csv(logpath, mode="a", header=False)
            with_duplicates = read_csv(logpath, index_col=0)
            without_duplicates = with_duplicates.drop_duplicates(
                ignore_index=True
            )
            self.dataframe = without_duplicates
            self.dataframe.to_csv(logpath)
        print("Results saved in log!")


class Test_Combination:
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
        self.sample_size = train_data.shape[0]
        X_train, y_train = preprocessing.encode_and_split(
            train=train_data, target=target, encoder=self.encoder
        )

        if self.interactions:
            X_train = self.interactions.fit_transform(X_train)
            X_train = DataFrame(X_train)

        if self.scale:
            X_train = self.scale.fit_transform(X_train)

        start_time = timer()

        self.cvs = _cv_mse(self.regressor, X_train, y_train)

        end_time = timer()

        self.time = end_time - start_time

        self.been_run = True
        self.log_results(log)
        self.print_summary()

    def log_results(self, log):
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
