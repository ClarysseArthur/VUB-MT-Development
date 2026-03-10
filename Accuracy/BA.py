import numpy as np 

def mh_binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Compute binary accuracy for multi-hot encoded vectors.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        Ground-truth binary labels.
    y_pred : array-like of shape (n_samples, n_labels)
        Predicted probabilities or scores.
    threshold : float, default=0.5
        Threshold used to convert predictions to 0/1.

    Returns
    -------
    float
        Binary accuracy over all label positions.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)

    y_pred_bin = (y_pred >= threshold).astype(int)
    return (y_true == y_pred_bin).mean()

def oh_binary_accuracy(y_true, y_pred):
    """
    Compute binary accuracy for one-hot encoded vectors.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes)
        Ground-truth one-hot encoded labels.
    y_pred : array-like of shape (n_samples, n_classes)
        Predicted probabilities or scores.

    Returns
    -------
    float
        Binary accuracy over all samples.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)

    y_pred_bin = np.zeros_like(y_pred)
    y_pred_bin[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1

    return (y_true == y_pred_bin).all(axis=1).mean()