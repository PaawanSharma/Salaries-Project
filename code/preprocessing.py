def X_y_split(dataframe, target=None):
    if not target:
        dataframe = dataframe.copy()
        X = dataframe.iloc[:, :-1]
        y = dataframe.iloc[:, -1]
    else:
        y = dataframe[target].copy()
        X = dataframe.drop(target, axis=1)
    return X, y


def encode(encoder, train, test):
    encoder.fit(train)
    new_train = encoder.transform(train)
    new_test = encoder.transform(test)
    return new_train, new_test


def encode_and_split(train, test, target=None, encoder=None):
    if encoder:
        train, test = encode(encoder, train, test)
    X_train, y_train = X_y_split(train, target)
    return X_train, y_train, test
