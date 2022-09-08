# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer

from typing import Optional, Union
import classes.layers.Layer
import numpy as np

class InputLayer(classes.layers.Layer.Layer):
    """
    Input layer
    """
    # Layer info
    name = None
    input = None
    output = None
    algorithm:str = None
    input_shape:Union[int, tuple, list] = None
    ouput_shape:Union[int, tuple, list] = None
    
    def __init__(self, ):
        """
        Class constructor
        """
        pass


    def calculate(self, input):
        """
        ...
        """
        pass

    def update(self):
        """
        ! IGNORE !
        Does not exist for this layer
        """
        pass