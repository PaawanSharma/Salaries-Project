"""Plotting functions for EDA."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_target(target, dataframe, hist_bins=20):
    """Plot a seaborn boxplot and histogram for the target variable.

    Keyword Arguments:
    -----------------
    target -- the target variable
    dataframe -- the pandas DataFrame containing the data
    hist_bins -- the number of bins to use in the histogram plot
    """
    target_fig = plt.figure(figsize=(16, 10))

    target_box = target_fig.add_subplot(2, 1, 1)
    sns.boxplot(dataframe[target],
                ax=target_box)

    target_hist = target_fig.add_subplot(2, 1, 2)
    sns.histplot(dataframe[target],
                 bins=hist_bins,
                 kde=True,
                 ax=target_hist)


def plot_categorical(feature, target, dataframe):
    """Plot a seaborn countplot and boxplot for the feature against the target.

    Keyword Arguments:
    -----------------
    feature -- the categorical variable under investigation
    target -- the target variable
    dataframe -- the pandas DataFrame containing the data
    """
    cat_fig = plt.figure(figsize=(21, 7))
    ascending = dataframe[target].groupby(
        dataframe[feature]).median().sort_values().index

    count_ax = cat_fig.add_subplot(1, 2, 1)
    cat_count = sns.countplot(dataframe[feature],
                              order=ascending,
                              ax=count_ax)

    box_ax = cat_fig.add_subplot(1, 2, 2)
    cat_box = sns.boxplot(x=feature,
                          y=target,
                          data=dataframe,
                          order=ascending,
                          ax=box_ax)

    if feature == 'companyId':
        cat_count.set_xticklabels([])
        cat_box.set_xticklabels([])
    else:
        cat_count.set_xticklabels(cat_count.get_xticklabels(), rotation=36)
        cat_box.set_xticklabels(cat_count.get_xticklabels(), rotation=36)
