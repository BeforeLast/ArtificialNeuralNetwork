# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

from typing import Optional, Union
import classes.layers.Layer
import numpy as np

class Dense(classes.layers.Layer.Layer):
    """
    Dense layer
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
        Calculate the given input tensor to given output
        """

        pass

    def update(self):
        """
        Update the layer's weight
        """
        pass