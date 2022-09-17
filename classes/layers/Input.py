# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer

from typing import Optional, Union
from classes.layers.Layer import Layer as BaseLayer
from classes.utils.ImageConvert import ImageConvert
import numpy as np

class InputLayer(BaseLayer):
    """
    Input layer
    """
    # Layer info
    name = None
    input = None
    output = None
    algorithm:str = None
    input_shape:Union[int, tuple, list] = None
    output_shape:Union[int, tuple, list] = None
    
    def __init__(self, input_shape, **kwargs):
        """
        Class constructor
        """
        if type(input_shape) is int:
            self.input_shape = (None, input_shape)
        elif type(input_shape) in [list, tuple, np.ndarray]:
            if input_shape[0] != None:
                # Only state data dimension and channel
                fix_shape = [None]
                fix_shape.extend(list(input_shape))
                self.input_shape = tuple(fix_shape)
            else:
                # Batch input shape already stated (None)
                self.input_shape = tuple(input_shape)
        else:
            raise TypeError(f'Type {type(input_shape)} cannot be used as input \
shape')
        self.name = kwargs.get("name", "InputLayer")

    def calculate(self, input) -> np.ndarray:
        """
        Check if input is the same as input_shape or not to prevent data shape
        mismatch through the model and convert input to np.ndarray
        """
        # Check input type
        if type(input) not in [list, tuple, np.ndarray]:
            raise TypeError(f"Unsuported input type for type {type(input)}")
        # Save input history
        self.input = input
        # Convert input type to np.ndarray
        output = np.array(input)
        # Check input shape
        if output.shape != self.input_shape[1:]:
            raise ValueError(f'Mismatched input shape, {self.input_shape[1:]} \
was expected and {output.shape} was given')
        # Save output history
        self.output = output.copy()
        return output

    def compile(self, input_shape=None):
        """
        COMPILING PURPOSE
        Compile layer with the given input shape
        """
        # Calculate output shape
        self.calculate_output_shape()

    def calculate_output_shape(self):
        """
        COMPILING PURPOSE
        Calculate ouput shape from layer's input shape
        """
        # Match output shape with input shape
        self.output_shape = self.input_shape

    def update(self):
        """
        ! IGNORE !
        Does not exist for this layer
        """
        pass

if __name__ == "__main__":
    input_test = InputLayer((28,28,3))
    input_test.compile()
    print("SHAPE TEST")
    print("input_shape:", input_test.input_shape)
    print("output_shape:", input_test.output_shape)
    print("\nPROCESS TEST")
    input_input = np.random.rand(28,28,3)
    input_output = input_test.calculate(input_input)
    print("process output shape:", input_output.shape)
    print("process output type:", type(input_output))