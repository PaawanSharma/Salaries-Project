from pandas import DataFrame, Series
from sklearn.model_selection import cross_val_score
from time import time as timer

import preprocessing


def cv_mse(estimator, X, y):
    return cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=True,
    )


class Log:
    def __init__(self):
        self.columns = [
            "Encoder",
            "Scaling",
            "Regressor",
            "CVS_1",
            "CVS_2",
            "CVS_3",
            "CVS_4",
            "CVS_5",
            "Time /s",
        ]
        self.dataframe = DataFrame(columns=self.columns)
        self.dataframe.index.name = "MODEL TEST"

    def add(self, encoder, scaling, regressor, cvs, time):
        row_data = [encoder, scaling, regressor] + cvs + [time]
        row = Series(row_data, self.columns)
        self.dataframe = self.dataframe.append(row, ignore_index=True)


class Test_Combination:
    def __init__(
        self, log_dataframe, encoder=None, scale=None, regressor=None
    ):
        self.encoder = encoder
        if self.encoder:
            self.encoder_name = self.encoder.__name__
        else:
            self.encoder_name = "NONE"
        self.scale = scale
        self.regressor = regressor
        self.been_run = False
        self.log_dataframe = log_dataframe

    def run(self, train_data, test_data, target):
        X_train, y_train, encoded_test = preprocessing.encode_and_split(
            train_data, test_data, target, self.encoder
        )
        start_time = timer()
        self.cvs = cv_mse(self.regressor, X_train, y_train)
        end_time = timer()
        self.time = end_time - start_time
        self.been_run = True
        self.log_results(self.log_dataframe)
        self.print_summary()

    def log_results(self, log_object):
        log_object.add(
            self.encoder_name,
            str(self.scale),
            str(self.regressor),
            self.cvs.tolist(),
            self.time,
        )
        print("Results added to log!")

    def print_summary(self):
        print(
            "Mean MSE for {} with {} encoding: {}".format(
                str(self.regressor), self.encoder_name, (self.cvs * -1).mean(),
            )
        )
