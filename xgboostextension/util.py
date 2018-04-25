import numpy as np


def _preprare_data_in_groups(X, y=None, sample_weights=None):
    """
    Takes the first column of the feature Matrix X given and
    transforms the data into groups accordingly.

    Parameters
    ----------
    X : (2d-array like) Feature matrix with the first column the group label

    y : (optional, 1d-array like) target values

    sample_weights : (optional, 1d-array like) sample weights

    Returns
    -------
    sizes: (1d-array) group sizes

    X_features : (2d-array) features sorted per group

    y : (None or 1d-array) Target sorted per group

    sample_weights: (None or 1d-array) sample weights sorted per group
    """
    group_labels = X[:,0]
    group_indices = group_labels.argsort()

    group_labels = group_labels[group_indices]
    _, sizes = np.unique(group_labels, return_counts=True)
    X = X[group_indices, 1:]

    if y is not None:
        y = y[group_indices]

    if sample_weights is not None:
        sample_weights = sample_weights[group_indices]

    return sizes, X, y, sample_weights
