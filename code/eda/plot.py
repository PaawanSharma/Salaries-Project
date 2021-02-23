"""Plot.

This module contains plotting functions for EDA.

This module requires that the Python environment contain installations of
plotting libraries matplotlib and seaborn.

This module contains four public functions:

    * plot_target - plot a boxplot and histogram for a target variable
    * plot_categorical - plot a countplot and boxplot for a feature vs a target
    * plot_numerical - plot count and hexbin plots for a feature vs a target
    * categorical_correlation - group data by category and plot correlation
    with mean target
"""

from matplotlib.pyplot import draw, figure, legend, subplots
from seaborn import boxplot, countplot, histplot, regplot


def plot_target(target, dataframe, hist_bins=20, target_label=None):
    """Plot a seaborn boxplot and histogram for the target variable.

    Keyword Arguments:
    -----------------
    target -- the target variable
    dataframe -- the pandas DataFrame containing the data
    hist_bins -- the number of bins to use in the histogram plot
    target_label -- a label for the target axis
    """
    # Create figure
    target_fig = figure(figsize=(16, 16))

    # Create boxplot
    box_ax = target_fig.add_subplot(2, 1, 1)
    target_box = boxplot(
        dataframe[target], flierprops={"markerfacecolor": "white"}, ax=box_ax
    )

    # Create histogram
    hist_ax = target_fig.add_subplot(2, 1, 2)
    target_hist = histplot(
        dataframe[target], bins=hist_bins, kde=True, ax=hist_ax
    )

    # Change the x-axis labels
    if target_label is not None:
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
    # Create figure
    cat_fig = figure(figsize=(21, 7))

    # Get order of category groups by target median
    ascending = (
        dataframe[target]
        .groupby(dataframe[feature])
        .median()
        .sort_values()
        .index
    )

    # Create countplot
    count_ax = cat_fig.add_subplot(1, 2, 1)
    cat_count = countplot(dataframe[feature], order=ascending, ax=count_ax)

    # Create boxplot
    box_ax = cat_fig.add_subplot(1, 2, 2)
    cat_box = boxplot(
        x=feature,
        y=target,
        data=dataframe,
        order=ascending,
        flierprops={"markerfacecolor": "white"},
        ax=box_ax,
    )

    # Stylise the plots for better visual clarity
    for plotted in [cat_count, cat_box]:
        _categoricals_axes(plotted, feature)


def plot_numerical(feature, target, dataframe, target_unit=None):
    """Plot seaborn count and hexbin plots for the feature against the target.

    Keyword Arguments:
    -----------------
    feature -- the categorical variable under investigation
    target -- the target variable
    dataframe -- the pandas DataFrame containing the data
    target_unit -- the units of the target variable (for axis labelling)
    """
    # Create figure
    fig = figure(figsize=(21, 7))

    # Create histogram
    hist_ax = fig.add_subplot(1, 2, 1)
    histplot(data=dataframe, x=feature, discrete=True, ax=hist_ax)

    # Create hexbin plot
    hex_ax = fig.add_subplot(1, 2, 2)
    hex_plot = dataframe.plot.hexbin(
        x=feature, y=target, gridsize=20, ax=hex_ax
    )

    # Add unit to hexbin plot's y-axis if unit given
    if target_unit is not None:
        hex_plot.set_ylabel("{} / {}".format(target.title(), target_unit))

    # Add lineplot showing the mean target value for each feature value
    mean_line = dataframe.groupby(feature)[target].mean()
    mean_line.plot(
        ax=hex_ax,
        label="mean {}/{}".format(target, feature),
        color="yellow",
        alpha=0.5,
    )

    # Display the legend for the lineplot
    legend()


def categorical_correlation(
    feature, target, dataframe, groupfunc, x_label=None, y_label=None
):
    """Group data by categorical feature and plot correlation with mean target.

    Keyword Arguments:
    -----------------
    feature -- the categorical feature to group by
    target -- the target variable
    dataframe -- the pandas DataFrame containing the data
    groupfunc -- the function to apply to the groups to quantify them
    x_label -- a label for the x-axis
    y_label -- a label for the y-axis
    """
    # Group the data by the feature values and apply a function to assign
    # groups numerical values
    group = dataframe.groupby(feature)
    grouped_feature = group.apply(groupfunc)

    # Get the mean value of the target for each group
    target_mean = group[target].mean()

    # Assign axis labels if specified
    if x_label is not None:
        grouped_feature.name = x_label
    if y_label is not None:
        target_mean.name = y_label

    # Create figure, axes and regplot
    cc_fig, cc_ax = subplots(figsize=(12, 12))
    reg_plot = regplot(x=grouped_feature, y=target_mean, marker=".")
    # Stylise the plot for better visual clarity
    _categoricals_axes(reg_plot)


def _categoricals_axes(plotted, feature=None):
    """Stylise a plot for improved visual clarity.

    Keyword Arguments:
    -----------------
    plotted -- the plot object
    feature -- the feature that was plotted
    """
    # draw must be called so that axes labels can be manipulated correctly
    draw()

    # Define rotation angle
    if feature == "jobType":
        rot_angle = 60
    else:
        rot_angle = 36

    # Rotate x-axis labels or remove them entirely in the case of companyID
    if feature == "companyId":
        plotted.set_xticklabels([])
    else:
        plotted.set_xticklabels(plotted.get_xticklabels(), rotation=rot_angle)

    # Add units to salary label
    if plotted.get_ylabel() == "salary":
        plotted.set_ylabel("Salary / 1000 USD")
