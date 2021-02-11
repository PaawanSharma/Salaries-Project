"""Plotting functions for EDA."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def plot_target(target, dataframe, hist_bins=20, target_label=None):
    """Plot a seaborn boxplot and histogram for the target variable.

    Keyword Arguments:
    -----------------
    target -- the target variable
    dataframe -- the pandas DataFrame containing the data
    hist_bins -- the number of bins to use in the histogram plot
    target_label -- a label for the target axis
    """
    target_fig = plt.figure(figsize=(16, 16))

    box_ax = target_fig.add_subplot(2, 1, 1)
    target_box = sns.boxplot(dataframe[target],
                flierprops={'markerfacecolor': 'white'},
                ax=box_ax)

    hist_ax = target_fig.add_subplot(2, 1, 2)
    target_hist = sns.histplot(dataframe[target],
                 bins=hist_bins,
                 kde=True,
                 ax=hist_ax)
    
    if target_label:
        for axes in [target_box, target_hist]:
            axes.set(xlabel=target_label)


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
                          flierprops={'markerfacecolor': 'white'},
                          ax=box_ax)

    for axes in [count_ax, box_ax]:
        categoricals_axes(axes, feature, "Salary / 1000 USD")

def categorical_correlation(feature, target, dataframe, groupfunc, x_label=None,
                            y_label = None):
    """Group data by categorical feature and plot correlation with mean target.
    
    Keyword Arguments:
    ----------------
    feature -- the categorical feature to group by
    target -- the target variable
    dataframe -- the pandas DataFrame containing the data
    groupfunc -- the function to use for grouping
    x_label -- a label for the x-axis
    y_label -- a label for the y-axis
    """
    group = dataframe.groupby(feature)
    grouped_feature = group.apply(groupfunc)
    
    if x_label:
        grouped_feature.name = x_label
    
    target_mean = group[target].mean()
    
    if y_label:
        target_mean.name = y_label
        
    cc_fig, cc_ax = plt.subplots(figsize=(12, 12))
    reg_plot = sns.regplot(x=grouped_feature, y=target_mean, marker='.')
    categoricals_axes(cc_ax)

def categoricals_axes(axes, feature=None, target_label=None):
    """Stylise plot axes for improved visual clarity.

    Keyword Arguments:
    -----------------
    axes -- the axes object
    feature -- the feature that was plotted
    target_label -- a label for the target axis
    """
    if feature == "companyId":
        ticks_loc = axes.get_xticks().tolist()
        axes.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        axes.set_xticklabels([])
    else:
        ticks_loc = axes.get_xticks().tolist()
        axes.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        axes.set_xticklabels(axes.get_xticklabels(), rotation=36)

    if target_label:
        axes.set(ylabel=target_label)
