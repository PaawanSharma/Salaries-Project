"""
Preprocessing.

This module contains three functions:

    * X_y_split - split a dataframe into features and a target
    * encode - return encoded train (and optionally test) dataframe
    * encode_and_split - perform encoding and X,y splitting
"""


def X_y_split(dataframe, target=None):
    """Split a dataframe into features and target.

    Keyword Arguments:
    -----------------
    dataframe -- the dataframe to be split
    target -- the target column; if None, the last column is used
    """
    if not target:
        dataframe = dataframe.copy()
        X = dataframe.iloc[:, :-1]
        y = dataframe.iloc[:, -1]
    else:
        y = dataframe[target].copy()
        X = dataframe.drop(target, axis=1)
    return X, y


def encode(encoder, train, test=None):
    """Apply encoding to training data and optionally test data.

    Keyword Arguments:
    -----------------
    encoder -- an encoder object from the feature_engineering.encoders module
    train -- the training dataframe
    test -- the test dataframe; if None, then only one dataframe is returned
    """
    # Fit the encoder
    encoder.fit(train)

    # Transform the training data
    new_train = encoder.transform(train)

    # Transform the test data if it exists and return all transformed data
    if test is not None:
        new_test = encoder.transform(test)
        return new_train, new_test
    return new_train


def encode_and_split(train, target, encoder=None, test=None):
    """Encode and split the training data and optionally test data.

    Keyword Arguments:
    -----------------
    train -- the training dataframe
    target -- the name of the target column for splitting and encoding
    encoder -- an encoder object from feature_engineering.encoders
    test -- the test dataframe; if None, then only one dataframe is returned
    """
    # Apply encoding if an encoder has been passed to the function
    if encoder is not None:
        if test is not None:
            train, test = encode(encoder, train, test)
        else:
            train = encode(encoder, train)

    # Split the features and target
    X_train, y_train = X_y_split(train, target)

    # Return split and encoded training set and encoded test set if it exists
    if test is not None:
        return X_train, y_train, test
    return X_train, y_train
