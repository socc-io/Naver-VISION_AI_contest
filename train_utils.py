import tensorflow as tf
import numpy as np


def l2_normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norm, where=norm!=0)  # only divide nonzeros else 1