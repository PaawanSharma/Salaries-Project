"""Statistical functions for EDA."""


def interquartile_rule(feature, dataframe):
    """Return outliers and counts of upper and lower outliers.

    Keyword Arguments:
    -----------------
    feature -- the numerical variable to apply the IQ rule to
    dataframe -- the pandas DataFrame containing the data
    """
    upper_quartile = dataframe[feature].quantile(0.75)
    lower_quartile = dataframe[feature].quantile(0.25)
    iqr = upper_quartile - lower_quartile
    upper_bound = upper_quartile + iqr * 1.5
    lower_bound = lower_quartile - iqr * 1.5

    outliers = dataframe[
        (dataframe[feature] < lower_bound) | (dataframe[feature] > upper_bound)
    ]

    upper_count = outliers[outliers[feature] > upper_bound].shape[0]
    lower_count = outliers[outliers[feature] < lower_bound].shape[0]

    return outliers, upper_count, lower_count
