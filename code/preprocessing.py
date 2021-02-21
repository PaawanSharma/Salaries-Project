def X_y_split(dataframe, target=None):
    if not target:
        dataframe = dataframe.copy()
        X = dataframe.iloc[:, :-1]
        y = dataframe.iloc[:, -1]
    else:
        y = dataframe[target].copy()
        X = dataframe.drop(target, axis=1)
    return X, y


def encode(encoder, train, test= None):
    encoder.fit(train)
    new_train = encoder.transform(train)
    if test:
        new_test = encoder.transform(test)
        return new_train, new_test
    return new_train


def encode_and_split(train, test=None, target=None, encoder=None):
    if encoder:
        if test:
            train, test = encode(encoder, train, test)
        else:
            train = encode(encoder, train)
    X_train, y_train = X_y_split(train, target)
    if test:
        return X_train, y_train, test
    return X_train, y_train