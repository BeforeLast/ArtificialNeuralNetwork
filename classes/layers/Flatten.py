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
    input_shape:tuple = None
    output_shape:tuple[None, int] = None
    
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
        self.output_shape = (None, np.prod(self.input_shape[1:]).item())

    def update(self):
        """
        ! IGNORE !
        Does not exist for this layer
        """
        pass
    
    def to_object(self):
        """
        SAVING/LOADING PURPOSE
        Convert self to json-like object (dictionary)
        """
        obj = {}
        obj['layer_type'] = 'flatten'
        obj['data'] = {}
        # Layer info
        obj['data']['name'] = self.name
        obj['data']['algorithm'] = self.algorithm
        obj['data']['input_shape'] = self.input_shape
        obj['data']['output_shape'] = self.output_shape
        return obj

    
    def from_object(self, object):
        """
        SAVING/LOADING PURPOSE
        Convert json-like object (dictionary) to layer object
        """
        # Layer info
        self.name = object['name']
        self.algorithm = object['algorithm']
        self.input_shape = tuple(object['input_shape'])
        self.output_shape = tuple(object['output_shape'])
        

if __name__ == "__main__":
    flatten_test = Flatten()
    input_shape = (25, 25, 16)
    expected_input_shape = (None, 25, 25, 16)
    expected_output_shape = (None, 25*25*16)
    flatten_test.compile(input_shape)
    print("input_shape TEST:", flatten_test.input_shape == expected_input_shape)
    print("output_shape TEST:", flatten_test.output_shape == expected_output_shape)