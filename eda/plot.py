"""Plotting functions for EDA."""

import matplotlib.pyplot as plt
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

    for plotted in [cat_count, cat_box]:
        categoricals_axes(plotted, feature, "Salary / 1000 USD")

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
    categoricals_axes(reg_plot)

def categoricals_axes(plotted, feature=None, target_label=None):
    """Stylise plot axes for improved visual clarity.

    Keyword Arguments:
    -----------------
    plotted -- the plot object
    feature -- the feature that was plotted
    target_label -- a label for the target axis
    """
    plt.draw()

    if feature == 'jobType':
        rot_angle = 60
    else:
        rot_angle = 36
        
    if feature == "companyId":
        plotted.set_xticklabels([])
    else:
        plotted.set_xticklabels(plotted.get_xticklabels(), rotation=rot_angle)

    if target_label:
        plotted.set(ylabel=target_label)
