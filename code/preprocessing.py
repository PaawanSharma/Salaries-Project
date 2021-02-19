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

def encode_and_split(encoder, train, test, target=None):
    encoded_train, encoded_test = encode(encoder, train, test)
    X_train, y_train = X_y_split(encoded_train, target)
    return X_train, y_train, encoded_test