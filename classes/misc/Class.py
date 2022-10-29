## LAYER IMPORTS
from classes.layers.Conv2D import Conv2D
from classes.layers.Dense import Dense
from classes.layers.Flatten import Flatten
from classes.layers.Input import InputLayer
from classes.layers.LSTM import LSTM
## LAYER CLASS PACK
layer_cpack = {
    'conv2d': Conv2D,
    'dense':Dense,
    'flatten':Flatten,
    'input':InputLayer,
    'lstm':LSTM
}

## LAYER DEFAULT CLASS INIT PACK
def conv2d_default():
    """
    Return default conv2d class
    """
    return_class = Conv2D(1, (2,2))
    return return_class

def dense_default():
    """
    Return default conv2d class
    """
    return_class = Dense(1)
    return return_class

def flatten_default():
    """
    Return default conv2d class
    """
    return_class = Flatten()
    return return_class

def input_default():
    """
    Return default conv2d class
    """
    return_class = InputLayer((None, 1))
    return return_class

def lstm_default():
    """
    Return default conv2d class
    """
    return_class = LSTM(1)
    return return_class

layerinit_cpack = {
    'conv2d': conv2d_default,
    'dense':dense_default,
    'flatten':flatten_default,
    'input':input_default,
    'lstm':lstm_default
}

## LOSS IMPORTS
from classes.losses.BinaryCrossentropy import BinaryCrossentropy

## LOSS CLASS PACK
loss_cpack = {
    'binary_crossentropy': BinaryCrossentropy,
}

## LOSS DEFAULT CLASS INIT PACK
def binary_crossentropy_default():
    """Return default binary crossentropy loss class"""
    return_class = BinaryCrossentropy()
    return return_class

lossinit_cpack = {
    'binary_crossentropy':binary_crossentropy_default
}