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
    output_shape:Union[int, tuple, list] = None
    
    def __init__(self, **kwargs):
        """
        Class constructor
        
        """
        self.name = kwargs.get("name", "FlattenLayer")

    def calculate(self, input:np.ndarray):
        """
        Convert multi-dimensional input tensors into a single dimension
        """
        return input.flatten()

    def compile(self, input_shape):
        """
        COMPILING PURPOSE
        Compile layer to be used by calucating output shape and assigning input
        shape
        """
        if input_shape[0] != None:
            # Only state data dimension and channel
            fix_shape = [None]
            fix_shape.extend(list(input_shape))
            self.input_shape = tuple(fix_shape)
        else:
            # Batch input shape already stated (None)
            self.input_shape = tuple(input_shape)
        self.calculate_output_shape()
            
    def calculate_output_shape(self):
        """
        COMPILING PURPOSE
        Calculate ouput shape from layer's input shape
        """
        self.output_shape = (None, np.prod(self.input_shape[1:]))

    def update(self):
        """
        ! IGNORE !
        Does not exist for this layer
        """
        pass

if __name__ == "__main__":
    flatten_test = Flatten()
    input_shape = (25, 25, 16)
    expected_input_shape = (None, 25, 25, 16)
    expected_output_shape = (None, 25*25*16)
    flatten_test.compile(input_shape)
    print("input_shape TEST:", flatten_test.input_shape == expected_input_shape)
    print("output_shape TEST:", flatten_test.output_shape == expected_output_shape)