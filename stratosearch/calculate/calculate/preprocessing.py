import numpy as np


def default_preprocess(arr: np.ndarray) -> np.ndarray:
    return arr.reshape((1, 1, arr.shape[0], arr.shape[1]))


PREPROCESSORS = {
}
