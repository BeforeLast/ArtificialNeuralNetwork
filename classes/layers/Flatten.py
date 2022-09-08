# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten

from typing import Optional, Union
import classes.layers.Layer
import numpy as np

class Flatten(classes.layers.Layer.Layer):
    """
    Flatten layer
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
        Convert multi-dimensional input tensors into a single dimension
        """
        pass

    def update(self):
        """
        ! IGNORE !
        Does not exist for this layer
        """
        pass