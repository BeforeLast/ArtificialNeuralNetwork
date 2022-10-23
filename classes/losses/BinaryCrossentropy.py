import numpy as np
from classes.losses.Losses import Losses

class BinaryCrossentropy(Losses):
    """
    BinaryCrossentropy loss class
    """
    def __init__(self) -> None:
        pass

    def calculate(self, target, prediction):
        """
        Calculate the losses for the given prediction(s) against target(s)
        using Binary Crossentropy
        """
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        a = np.log(1-prediction + 1e-7) - np.log(1-prediction + 1e-7) * target
        b = target * np.log(prediction + 1e-7)
        return -np.mean(a+b, axis=0)