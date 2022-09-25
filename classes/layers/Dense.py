# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

from typing import Optional, Union
from classes.layers.Layer import Layer as BaseLayer
from classes.misc.Function import dense_fpack
import numpy as np

class Dense(BaseLayer):
    """
    Dense layer
    """
    # Layer info
    name = None
    input = None
    output = None
    algorithm:str = None
    input_shape:tuple[None, int] = None
    output_shape:tuple[None, int] = None
    num_of_units:int = None
    weights:np.ndarray = None
    
    def __init__(self, units, activation='relu', **kwargs):
        """
        Class constructor
        """
        # Number of unit
        if type(units) is int:
            if units < 1:
                raise ValueError('Number of unit must be an integer greater than zero')
            else:
                self.num_of_units = units
        else:
            raise TypeError('Number of unit must be an integer')
        
        # Activation algorithm
        if type(activation) is not str:
            raise TypeError('Activation algorithm must be a string')
        elif activation.lower() not in ['relu', 'sigmoid']:
            raise NotImplementedError('Activation algorithm is not supported')
        else:
            self.algorithm = activation.lower()
        
        # Misc
        self.name = kwargs.get("name", "Dense")

    def calculate(self, input):
        """
        Calculate the given input tensor to given output
        """
        # Check input type
        if type(input) is not list \
            and type(input) is not tuple \
            and type(input) is not np.ndarray:
            # Input is not a list/tuple/np.ndarray
            raise TypeError('Input must be a list or a tuple')
        elif np.array(input).shape != self.input_shape[1:]:
            # Input shape mismatch
            raise ValueError(f'Expected {self.input_shape[1:]} shape \
but {np.array(input).shape} shape was given.')
        else:
            # Convert input to np.ndarray
            self.input = np.array(input)
            # Add bias value at the beginning of the input (not bias weight)
            self.input = np.insert(self.input, 0 , values=1)
            # Dot product
            dot_prod = np.dot(self.input, self.weights)
            # Apply activation function to dot product
            output = dense_fpack[self.algorithm](dot_prod)
            self.output = output.copy()
            return output

    # ANCHOR : COMPILING
    def compile(self, input_shape):
        """
        COMPILING PURPOSE
        Compile layer with the given input shape
        """
        # Configure layer's class input_shape
        if type(input_shape) is int:
            # Only state data dimension
            self.input_shape = (None, input_shape)
        elif type(input_shape) is tuple or type(input_shape) is list:
            # Input_shape is in a form of tuple or list
            if len(input_shape) == 2:
                # Batch (None) is stated and input size is stated
                self.input_shape = tuple(input_shape)
            elif len(input_shape) == 1:
                # Only input size is stated
                self.input_shape = (None, input_shape[0])
        # Configure layer's class output_shape
        self.calculate_output_shape()
        # Instantiate weights
        self.generate_weights()

    def generate_weights(self):
        """
        COMPILING PURPOSE
        Generate weigths matrix from current input_shape
        """
        # Only generate weight if it is not generated yet
        if self.weights is None:
            self.weights = np.random.rand(
                self.input_shape[-1] + 1,
                self.num_of_units
            )

    def calculate_output_shape(self):
        """
        COMPILING PURPOSE
        Calculate ouput shape from layer's input shape
        """
        self.output_shape = (None, self.num_of_units)

    def update(self):
        """
        Update the layer's weight
        """
        pass

    def to_object(self):
        """
        SAVING/LOADING PURPOSE
        Convert self to json-like object (dictionary)
        """
        obj = {}
        obj['layer_type'] = 'dense'
        obj['data'] = {}
        # Layer info
        obj['data']['name'] = self.name
        obj['data']['algorithm'] = self.algorithm
        obj['data']['input_shape'] = self.input_shape
        obj['data']['output_shape'] = self.output_shape
        obj['data']['num_of_units'] = self.num_of_units
        obj['data']['weights'] = self.weights.tolist()
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
        self.num_of_units = object['num_of_units']
        self.weights = np.array(object['weights'])


if __name__ == "__main__":
    dense_test = Dense(2)
    dense_test.compile((None, 8))
    dense_input = np.arange(8)
    print("input:", dense_input)
    print(dense_test.weights)
    print(dense_test.weights.shape)
    print()
    print()
    dense_output = dense_test.calculate(dense_input)
    print(dense_output)