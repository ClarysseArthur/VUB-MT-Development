import numpy as np 

def mh_binary_accuracy(y_true, y_pred, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)

    y_pred_bin = (y_pred >= threshold).astype(int)
    return (y_true == y_pred_bin).mean()

def oh_binary_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)

    y_pred_bin = np.zeros_like(y_pred)
    y_pred_bin[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1

    return (y_true == y_pred_bin).all(axis=1).mean()