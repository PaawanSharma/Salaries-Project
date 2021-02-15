from pandas import get_dummies

def dummify(features, dataframe, exclude):
    excluded = dataframe.drop(exclude, axis=1)
    features = features.copy()
    for feature in exclude:
        try:
            features.remove(feature)
        except:
            pass
            
    
    dummied = get_dummies(excluded,
                          prefix=features,
                          columns=features,
                          drop_first=True)
    return dummied


class NotUniqueError(Exception):
    pass


def ordinal_encode(features, target, metric, dataframe):
    
    dataframe = dataframe.copy()
    
    if len(features) == 0:
        return dataframe
    
    feature = features[0]
    
    metric_values = dataframe.groupby(feature)[target].apply(metric).sort_values()
    
    if not metric_values.is_unique:
        raise NotUniqueError("Ordinal encoding cannot be executed with these \
                             arguments as two or more groups of {} have the same \
                                 {} {}.".format(feature,
                                                target,
                                                metric.__name__
                                                ))
    
    order = metric_values.index
    
    dataframe[feature] = dataframe[feature].replace(order, range(len(order)))
    
    return ordinal_encode(features[1:], target, metric, dataframe)