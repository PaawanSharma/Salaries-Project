"""Encoding functions for categorical variables."""

from pandas import get_dummies


def dummify(features, dataframe, exclude):
    """Apply dummy coding.

    Keyword Arguments:
    -----------------
    features -- a list of categorical features for encoding
    dataframe -- the pandas DataFrame containing the data
    exclude -- features to drop from the output
    """
    excluded = dataframe.drop(exclude, axis=1)
    features = features.copy()
    for feature in exclude:
        try:
            features.remove(feature)
        except ValueError:
            pass

    dummied = get_dummies(excluded,
                          prefix=features,
                          columns=features,
                          drop_first=True)
    return dummied


class NotUniqueError(Exception):
    """Exception for when a series isn't unique."""

    pass


def ordinal_encode(features, dataframe, metric, target):
    """Apply ordinal encoding.

    Keyword Arguments:
    -----------------
    features -- a list of categorical features for encoding
    target -- the target variable
    metric -- the function for evaluating the target in relation to each group
    dataframe -- the pandas DataFrame containing the data
    """
    dataframe = dataframe.copy()

    if len(features) == 0:
        return dataframe

    feature = features[0]

    metric_values = dataframe.groupby(feature)[target].\
        apply(metric).sort_values()

    if not metric_values.is_unique:
        raise NotUniqueError("Ordinal encoding cannot be executed with \
                             these arguments as two or more groups of {} \
                                 have the same {} {}.".format(feature,
                                                              target,
                                                              metric.__name__))

    order = metric_values.index
    dataframe[feature] = dataframe[feature].replace(order, range(len(order)))

    return ordinal_encode(features[1:], dataframe, metric, target)
