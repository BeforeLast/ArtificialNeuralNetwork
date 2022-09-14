# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

from typing import Optional, Union
from classes.layers.Layer import Layer as BaseLayer
import numpy as np

def relu(num):
    return np.maximum(num, 0)
    
def sigmoid(num):
    return 1 / (1 + np.exp(-num))

class Dense(BaseLayer):
    """
    Dense layer
    """
    # Layer info
    name = None
    input = None
    output = None
    algorithm:str = None
    input_shape:Union[int, tuple, list] = None
    number_unit:int = None
    ouput_shape:Union[int, tuple, list] = None
    weights = None
    biases = None
    
    def __init__(self, 
                 input_shape,
                 number_unit, 
                 algorithm='relu'
                ):
        """
        Class constructor
        """
        if type(input_shape) is int:
            if input_shape < 1:
                raise ValueError('Input shape must be an integer not less than one')
            else:
                self.input_shape = (input_shape,)
        elif (
            (type(input_shape) is tuple or type(input_shape) is list)
        ):
            if(len(input_shape) == 2
            and type(input_shape[0]) is int 
            and type(input_shape[1]) is int):
                if input_shape[0] < 1 or input_shape[1] != 1:
                    raise ValueError('Input size must be an integer not less than one and the second value must be one')
                else:
                    self.input_shape = tuple(input_shape[0])
            elif(len(input_shape) == 1
                and type(input_shape[0]) is int):
                if input_shape[0] < 1:
                    raise ValueError('Input size must be an integer not less than one')
                else:
                    self.input_shape = tuple(input_shape)
        else:
            raise ValueError('Input shape must be an integer, a list, or a tuple')
            
        # Number of unit
        if type(number_unit) is int:
            if number_unit < 1:
                raise ValueError('Number of unit must be an integer greater than zero')
            else:
                self.number_unit = number_unit
        else:
            raise ValueError('Number of unit must be an integer')
        
        # Activation algorithm
        if type(algorithm) is not str:
            raise ValueError('Activation algorithm must be a string')
        elif algorithm.lower() not in ['relu', 'sigmoid']:
            raise ValueError('Activation algorithm is not supported')
        else:
            self.algorithm = algorithm
            
        # Weights and biases
        self.weights = np.random.rand(self.number_unit, self.input_shape[0])
        self.biases = np.random.rand(self.number_unit)

    def calculate(self, input):
        """
        Calculate the given input tensor to given output
        """
        if type(input) is not list and type(input) is not tuple:
            raise ValueError('Input must be a list or a tuple')
        else:
            self.input = np.asarray(input)
            output_unactivated = np.zeros(self.number_unit)
            for i in range(self.number_unit):
                output_unactivated[i] = self.input.dot(self.weights[i]) + self.biases[i]
            if self.algorithm.lower() == 'relu':
                self.output = relu(output_unactivated)
            elif self.algorithm.lower() == 'sigmoid':
                self.output = sigmoid(output_unactivated)
            return self.output
            

    def update(self):
        """
        Update the layer's weight
        """
        pass