# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM

from classes.layers.Layer import Layer as BaseLayer
import numpy as np

class LSTM(BaseLayer):
    """
    Long Short-Term Memory (LSTM) layer
    """
    # Layer info
    name = None
    input = None
    output = None
    algorithm:str = None
    num_of_units:int = None
    input_shape:tuple = None
    output_shape:tuple = None
    return_sequences:bool = None

    # Weights
    U_forget:np.array = None
    U_input:np.array = None
    U_cell:np.array = None
    U_output:np.array = None

    W_forget:np.array = None
    W_input:np.array = None
    W_cell:np.array = None
    W_output:np.array = None

    b_forget:float = None
    b_input:float = None
    b_cell:float = None
    b_output:float = None

    # Connectivity
    ht:np.array = None
    ht_history:np.array = None
    ct:np.array = None
    ct_history:np.array = None
    
    def __init__(self, units, activation='relu', return_sequences=False, **kwargs):
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
        
        # Return sequences
        if type(return_sequences) is not bool:
            raise TypeError('Return sequences parameter must be a boolean')
        else:
            self.return_sequences = return_sequences
        
        # Misc
        self.name = kwargs.get("name", "LSTM")

    def calculate(self, input):
        """
        TODO
        """
        self.output = None
        return self.output

    # ANCHOR : COMPILING
    def compile(self, input_shape):
        """
        COMPILING PURPOSE
        Compile layer with the given input shape
        """
        # Configure layer's class input_shape
        if type(input_shape) is not tuple and type(input_shape) is not list:
            # Check input_shape type
            raise TypeError("Unknown input_shape type")
        else:
            # Check input_shape shape
            if len(input_shape) != 3:
                # Input shape dimension is invalid
                # expected ndim=3
                raise ValueError(
                    f"Expected ndim=3, found ndim={len(input_shape)}")
            else:
                # Input shape dimension is valid (ndim=3)
                if input_shape[0] is not None:
                    # ndim=3 expected to have None in the first dimension
                    # as a batch dimension
                    raise ValueError(
                        f"Expected ndim=3, found ndim={len(input_shape)+1}")
                else:
                    # None exist in first dimension of ndim=3
                    # convert input_shape to tuple
                    self.input_shape = tuple(input_shape)
        # Generate weights
        self.generate_weights()
        # Calculate output shape
        self.calculate_output_shape()

    def generate_weights(self):
        """
        COMPILING PURPOSE
        Generate weigths matrix from current input_shape
        """
        # Only generate weight if it is not generated yet
        if self.U_forget is None:
            self.U_forget = np.random.rand(self.num_of_units, self.input_shape[-1])
        if self.U_input is None:
            self.U_input = np.random.rand(self.num_of_units, self.input_shape[-1])
        if self.U_cell is None:
            self.U_cell = np.random.rand(self.num_of_units, self.input_shape[-1])
        if self.U_output is None:
            self.U_output = np.random.rand(self.num_of_units, self.input_shape[-1])
        
        if self.W_forget is None:
            self.W_forget = np.random.rand(self.num_of_units, self.num_of_units)
        if self.W_input is None:
            self.W_input = np.random.rand(self.num_of_units, self.num_of_units)
        if self.W_cell is None:
            self.W_cell = np.random.rand(self.num_of_units, self.num_of_units)
        if self.W_output is None:
            self.W_output = np.random.rand(self.num_of_units, self.num_of_units)
        
        if self.b_forget is None:
            self.b_forget = np.random.rand(self.num_of_units, 1)
        if self.b_input is None:
            self.b_input = np.random.rand(self.num_of_units, 1)
        if self.b_cell is None:
            self.b_cell = np.random.rand(self.num_of_units, 1)
        if self.b_output is None:
            self.b_output = np.random.rand(self.num_of_units, 1)

    def calculate_output_shape(self):
        """
        COMPILING PURPOSE
        Calculate ouput shape from layer's input shape
        """
        # Match output shape with input shape
        if self.return_sequences:
            self.output_shape = (None, self.input_shape[-2], self.num_of_units)
        else:
            self.output_shape = (None, self.num_of_units)

    def backward(self, next_layer = None, target = None):
        """
        ! IGNORE !
        Not required yet
        """
        pass

    def update(self, learning_rate):
        """
        ! IGNORE !
        Not required yet
        """
        pass

    def to_object(self):
        """
        TODO
        SAVING/LOADING PURPOSE
        Convert self to json-like object (dictionary)
        """
        obj = {}
        obj['layer_type'] = 'lstm'
        obj['data'] = {}
        
        # Layer info
        obj['data']['name'] = self.name
        obj['data']['algorithm'] = self.algorithm
        obj['data']['input_shape'] = self.input_shape
        obj['data']['output_shape'] = self.output_shape
        return obj
    
    def from_object(self, object):
        """
        TODO
        SAVING/LOADING PURPOSE
        Convert json-like object (dictionary) to layer object
        """
        # Layer info
        self.name = object['name']
        self.algorithm = object['algorithm']
        self.input_shape = tuple(object['input_shape'])
        self.output_shape = tuple(object['output_shape'])

if __name__ == "__main__":
    pass