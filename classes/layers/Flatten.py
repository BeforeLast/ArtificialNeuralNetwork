# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten

from typing import Optional, Union
from classes.layers.Layer import Layer as BaseLayer
import numpy as np

class Flatten(BaseLayer):
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
    flatten_size:int = None
    
    def __init__(self, input_shape):
        """
        Class constructor
        """
        if type(input_shape) is int:
            if input_shape < 1:
                raise ValueError('Input shape must be an integer not less than one')
            else:
                self.input_shape = (input_shape, input_shape)
                self.size = input_shape * input_shape
        elif (
            (type(input_shape) is tuple or type(input_shape) is list)
            and len(input_shape) == 2
            and type(input_shape[0]) is int 
            and type(input_shape[1]) is int
        ):
            if input_shape[0] < 1 or input_shape[1] < 1:
                raise ValueError('Input size must be an integer not less than one')
            else:
                self.input_shape = input_shape
                self.size = input_shape[0] * input_shape[1]

    def calculate(self, input):
        """
        Convert multi-dimensional input tensors into a single dimension
        """
        if type(input) is not list and type(input) is not tuple:
            raise ValueError('Input must be a list or a tuple')
        else:
            calculate_input_shape = np.shape(input)
            acceptable_input_shape = tuple(self.input_shape)
            if calculate_input_shape != acceptable_input_shape:
                raise ValueError('Input shape is not compatible. The input shape is {calculate_input_shape} \
                                 but the acceptable size is {acceptable_input_shape}')
            else:
                flatted_layer = []
                for i in range(len(input)):
                    for j in range(len(input[0])):
                        flatted_layer.append(input[i][j])
                return flatted_layer

    def update(self):
        """
        ! IGNORE !
        Does not exist for this layer
        """
        pass