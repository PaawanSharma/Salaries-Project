from matplotlib.pyplot import figure, show
from numpy import where, zeros_like, triu_indices_from
from seaborn import set_context, heatmap
from sklearn.exceptions import NotFittedError

from exceptions import NotUniqueError


class Encoder:
    def fit(self, dataframe, *args, **kwargs):
        self._exclusion_and_features(dataframe)
        self._fit(dataframe)
        self.fitted = True

    def transform(self, dataframe):
        if not self.fitted:
            raise NotFittedError
        return self._transform(dataframe)

    def correlation_matrix(self, dataframe):
        return self.transform(dataframe).corr()

    def reordered_correlation_matrix(self, dataframe, reordering):
        original_matrix = self.correlation_matrix(dataframe)
        new_index = original_matrix.columns[reordering]
        reordered_matrix = original_matrix.reindex(
            index=new_index, columns=new_index
        )
        return reordered_matrix

    def correlation_map(
        self, dataframe, cmap, size, fontsize, dp, reordering=None
    ):
        if reordering:
            matrix = self.reordered_correlation_matrix(dataframe, reordering)
        else:
            matrix = self.correlation_matrix(dataframe)
        mask = zeros_like(matrix)
        mask[triu_indices_from(mask)] = 1
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
        """Generate list of categorical features"""
        if not self.features:
            self.features = dataframe.columns[
                dataframe.dtypes == "object"
            ].tolist()
        for feature in self.exclude:
            try:
                self.features.remove(feature)
            except ValueError:
                pass

    def _exclusion_and_features(self, dataframe):
        self._apply_exclusion_to_df(dataframe)
        self._set_features(dataframe)

    def _apply_exclusion_to_df(self, dataframe):
        """Return dataframe without excluded columns"""
        return dataframe.drop(self.exclude, axis=1, errors="ignore")


class Group_Encoder(Encoder):
    def __init__(self, metric, target, features=[], exclude=[]):
        self.metric = metric
        self.target = target
        self.features = []
        self.exclude = exclude
        self.mapping = {}
        self.fitted = False

    def _fit(self, dataframe):
        for feature in self.features:
            metric_values = (
                dataframe.groupby(feature)[self.target]
                .apply(self.metric)
                .sort_values()
            )
            if not metric_values.is_unique:
                raise NotUniqueError(
                    "Ordinal encoding cannot be fitted with these \
parameters as two or more groups of {} have the \
same {} {}.".format(
                        feature, self.target, self.metric.__name__
                    )
                )
            self._generate_mapping(feature, metric_values)

    def _transform(self, dataframe):
        dataframe = dataframe.copy()
        return self._transform_mechanics(dataframe)


class Ordinal_Encoder(Group_Encoder):
    def _generate_mapping(self, feature, metric_values):
        order = tuple(metric_values.index)
        self.mapping[order] = range(len(order))

    def _transform_mechanics(self, dataframe):
        i = 0
        for key, value in self.mapping.items():
            dataframe[self.features[i]] = dataframe[self.features[i]].replace(
                key, value
            )
            i += 1
        return dataframe


class Target_Encoder(Group_Encoder):
    def _generate_mapping(self, feature, metric_values):
        self.mapping[feature] = metric_values.to_dict()

    def _transform_mechanics(self, dataframe):
        for feature in self.features:
            dataframe[feature] = dataframe[feature].replace(
                self.mapping[feature]
            )
        return dataframe


class Dummy_Encoder(Encoder):
    def __init__(self, features=[], exclude=[]):
        self.features = features
        self.feature_levels = {}
        self.exclude = exclude
        self.fitted = False

    def _fit(self, dataframe):
        for feature in self.features:
            levels = dataframe[feature].unique().tolist()
            levels.sort()
            levels = [level for level in levels[1:]]
            self.feature_levels[feature] = levels

    def _transform(self, dataframe):
        dataframe = dataframe.copy()
        for feature, levels in self.feature_levels.items():
            for level in levels:
                dummy_data = where(dataframe[feature] == level, 1, 0)
                dummy_name = "_".join([feature, level])
                dataframe[dummy_name] = dummy_data.astype("u1")
            dataframe.drop(feature, axis=1, inplace=True)
        return dataframe
