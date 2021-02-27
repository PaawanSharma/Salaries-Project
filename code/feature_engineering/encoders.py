"""Encoders.

This module contains Encoder objects for encoding categorical features.

This module requires that the Python environment contain installations of
numpy, matplotlib, seaborn and sklearn.

This module contains three public classes:

    * Ordinal_Encoder - to assign integer values to categories by a ranking of
    some metric of the target values for each category
    * Target_Encoder - to assign the value of some metric of the target values
    for each category to categories
    * Dummy_Encoder - to apply dummy coding to categories
"""

# Third-party imports
from matplotlib.pyplot import figure, show
from numpy import where, zeros_like, triu_indices_from
from seaborn import set_context, heatmap
from sklearn.exceptions import NotFittedError

# Local imports
from exceptions import NotUniqueError


class _Encoder:
    """_Encoder is the base class for all encoder classes."""

    def fit(self, dataframe, *args, **kwargs):
        """Fit the encoder to the data."""
        # Get rid of excluded features and set features for encoder
        self._exclusion_and_features(dataframe)
        # Perform the fitting
        self._fit(dataframe)
        # Register that the encoder has been fitted
        self.fitted = True

    def transform(self, dataframe):
        """Transform data for a fitted encoder."""
        # Make sure encoder is fitted
        if not self.fitted:
            raise NotFittedError
        # Get rid of all excluded features from data to be transformed
        dataframe = self._apply_exclusion_to_df(dataframe)
        # Return the transformed dataframe
        return self._transform(dataframe)

    def correlation_matrix(self, dataframe):
        """Produce a correlation matrix for a dataframe if it were encoded."""
        return self.transform(dataframe).corr()

    def reordered_correlation_matrix(self, dataframe, reordering):
        """Produce a correlation matrix with a custom ordering of index."""
        # Get the correlation matrix
        original_matrix = self.correlation_matrix(dataframe)

        # Reorder the matrix
        new_index = original_matrix.columns[reordering]
        reordered_matrix = original_matrix.reindex(
            index=new_index, columns=new_index
        )

        return reordered_matrix

    def correlation_map(
        self, dataframe, cmap, size, fontsize, dp, reordering=None
    ):
        """Produce correlation plot for a dataframe if it were encoded."""
        # Get the correlation matrix
        if reordering:
            matrix = self.reordered_correlation_matrix(dataframe, reordering)
        else:
            matrix = self.correlation_matrix(dataframe)

        # Get a mask to cover redundant data in the plot
        mask = zeros_like(matrix)
        mask[triu_indices_from(mask)] = 1

        # Set aesthetical information and produce figure and heatmap
        set_context("poster", font_scale=fontsize)
        size = (size, size)
        fig = figure(figsize=size)
        heatmap(
            matrix,
            center=0,
            cmap=cmap,
            mask=mask,
            linewidths=0.5,
            linecolor="black",
            annot=True,
            fmt=".{}f".format(dp),
            cbar=False,
            figure=fig,
        )
        show()

    def _set_features(self, dataframe):
        """Generate list of categorical features."""
        # Automatically get features from data if list of features not passed
        if not self.features:
            self.features = dataframe.columns[
                dataframe.dtypes == "object"
            ].tolist()
        # Get rid of any features passed to exclude
        for feature in self.exclude:
            try:
                self.features.remove(feature)
            except ValueError:
                pass

    def _exclusion_and_features(self, dataframe):
        """Get rid of excluded features and set features for encoder."""
        dataframe = self._apply_exclusion_to_df(dataframe)
        self._set_features(dataframe)

    def _apply_exclusion_to_df(self, dataframe):
        """Return dataframe without excluded columns."""
        return dataframe.drop(self.exclude, axis=1, errors="ignore")


class _Group_Encoder(_Encoder):
    """_Group_Encoder is the base class for target and ordinal encoders."""

    def __init__(self, metric, target, features=[], exclude=[]):
        self.metric = metric
        self.target = target
        self.features = features
        self.exclude = exclude
        self.mapping = {}
        self.fitted = False

    def _fit(self, dataframe):
        """Perform the fitting."""
        # For each feature, group categories, calculate metric, order groups,
        # check for non-unique groups and produce a mapping for transformations
        # if no clashes exist
        for feature in self.features:
            # Group categories, calculate metric and order groups
            metric_values = (
                dataframe.groupby(feature)[self.target]
                .apply(self.metric)
                .sort_values()
            )
            # Raise exception if multiple groups have the same output
            if not metric_values.is_unique:
                raise NotUniqueError(
                    "Ordinal encoding cannot be fitted with these \
parameters as two or more groups of {} have the \
same {} {}.".format(
                        feature, self.target, self.metric.__name__
                    )
                )
            # Produce a mapping for transformations
            self._generate_mapping(feature, metric_values)

    def _transform(self, dataframe):
        """Return a transformed dataframe."""
        dataframe = dataframe.copy()
        return self._transform_mechanics(dataframe)

    def _transform_mechanics(self, dataframe):
        """Carry out the transformation."""
        for feature in self.features:
            dataframe[feature] = dataframe[feature].map(self.mapping[feature])
        return dataframe


class Ordinal_Encoder(_Group_Encoder):
    """Class for ordinal encoding of features.

    Encodes dataframe columns by replacing values with integers ordered based
    on values the target takes for all rows with each feature value.

    Instance variable are inherited from _Group_Encoder and public methods are
    inherited from _Encoder via _Group_Encoder.

    Instance variables
    ------------------
    metric -- the function used to score the target values for each group.
    target -- the target variable to be used for scoring groups.
    features -- a list of which features to encode; if None, all
    categorical features are encoded.
    exclude -- a list of any categorical features to drop from the dataset and
    not encode.

    Public methods
    --------------
    fit -- fit the encoder to a dataset.
    transform -- transform a dataset with a fitted encoder.
    correlation_matrix -- encode a dataset and produce a correlation matrix for
    the encoded data.
    reordered_correlation_matrix -- same as correlation_matrix but with the
    indices in a custom order.
    correlation_map -- encode a dataset and produce a plot of its correlation
    matrix.
    """

    def __init__(self, *args, **kwargs):
        # Inherit initiation function from super-class with some addiitions
        super(Ordinal_Encoder, self).__init__(*args, **kwargs)

        # Give encoder a __name__ based on input parameters
        if not self.features:
            feature_string_ref = "ALL"
        else:
            feature_string_ref = self.features
        if self.exclude:
            name_string = "Ordinal_Encoder(metric={}, target={}, features={}, \
exclude={})".format(
                self.metric.__name__,
                self.target,
                feature_string_ref,
                self.exclude,
            )
        else:
            name_string = "Ordinal_Encoder(metric={}, target={}, \
features={})".format(
                self.metric.__name__, self.target, feature_string_ref
            )

        self.__name__ = name_string

    def _generate_mapping(self, feature, metric_values):
        """Produce a mapping for a feature's transformation."""
        order = tuple(metric_values.index)
        self.mapping[feature] = dict(zip(order, range(len(order))))


class Target_Encoder(_Group_Encoder):
    """Class for target encoding of features.

    Encodes dataframe columns by replacing each value with a score calculated
    for the list of values the target takes for all rows with that feature
    value.

    Instance variable are inherited from _Group_Encoder and public methods are
    inherited from _Encoder via _Group_Encoder.

    Instance variables
    ------------------
    metric -- the function used to score the target values for each group.
    target -- the target variable to be used for scoring groups.
    features -- a list of which features to encode; if None, all
    categorical features are encoded.
    exclude -- a list of any categorical features to drop from the dataset and
    not encode.

    Public methods
    --------------
    fit -- fit the encoder to a dataset.
    transform -- transform a dataset with a fitted encoder.
    correlation_matrix -- encode a dataset and produce a correlation matrix for
    the encoded data.
    reordered_correlation_matrix -- same as correlation_matrix but with the
    indices in a custom order.
    correlation_map -- encode a dataset and produce a plot of its correlation
    matrix.
    """

    def __init__(self, *args, **kwargs):
        # Inherit initiation function from super-class with some addiitions
        super(Target_Encoder, self).__init__(*args, **kwargs)

        # Give encoder a name based on input parameters
        if not self.features:
            feature_string_ref = "ALL"
        else:
            feature_string_ref = self.features
        if self.exclude:
            name_string = "Target_Encoder(metric={}, target={}, features={}, \
exclude={})".format(
                self.metric.__name__,
                self.target,
                feature_string_ref,
                self.exclude,
            )
        else:
            name_string = "Target_Encoder(metric={}, target={}, \
features={})".format(
                self.metric.__name__, self.target, feature_string_ref
            )

        self.__name__ = name_string

    def _generate_mapping(self, feature, metric_values):
        """Produce a mapping for a feature's transformation."""
        self.mapping[feature] = metric_values.to_dict()


class Dummy_Encoder(_Encoder):
    """Class for target encoding of features.

    Encodes dataframe columns by creating a new column for each possible value
    and deletes one new column per encoded feature to avoid introducing new
    collinearities.

    Public methods are inherited from _Encoder via _Group_Encoder.

    Instance variables
    ------------------
    features -- a list of which features to encode; if None, all
    categorical features are encoded.
    exclude -- a list of any categorical features to drop from the dataset and
    not encode.

    Public methods
    --------------
    fit -- fit the encoder to a dataset.
    transform -- transform a dataset with a fitted encoder.
    correlation_matrix -- encode a dataset and produce a correlation matrix for
    the encoded data.
    reordered_correlation_matrix -- same as correlation_matrix but with the
    indices in a custom order.
    correlation_map -- encode a dataset and produce a plot of its correlation
    matrix.
    """

    def __init__(self, features=[], exclude=[]):
        self.features = features
        self.feature_levels = {}
        self.exclude = exclude
        self.fitted = False
        if features:
            feature_string_ref = self.features
        else:
            feature_string_ref = "ALL"
        if not exclude:
            exclude_string_ref = "NONE"
        else:
            exclude_string_ref = self.exclude
        self.__name__ = "Dummy_Encoder(features={}, exclude={})".format(
            feature_string_ref, exclude_string_ref
        )

    def _fit(self, dataframe):
        """Perform the fitting."""
        for feature in self.features:
            levels = dataframe[feature].unique().tolist()
            levels.sort()
            levels = [level for level in levels[1:]]
            self.feature_levels[feature] = levels

    def _transform(self, dataframe):
        """Return a transformed dataframe."""
        dataframe = dataframe.copy()
        for feature, levels in self.feature_levels.items():
            for level in levels:
                dummy_data = where(dataframe[feature] == level, 1, 0)
                dummy_name = "_".join([feature, level])
                dataframe[dummy_name] = dummy_data.astype("u1")
            dataframe.drop(feature, axis=1, inplace=True)
        return dataframe
