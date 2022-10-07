import numpy as np

class BinaryCrossEntropy :
  def BinaryCrossEntropy(target, pred):
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    a = np.log(1-pred + 1e-7) - np.log(1-pred + 1e-7) * target
    b = target * np.log(pred + 1e-7)
    return -np.mean(a+b, axis=0)