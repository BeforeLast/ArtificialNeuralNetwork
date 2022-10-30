# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM

from classes.layers.Layer import Layer as BaseLayer
import numpy as np
from classes.misc.Function import lstm_fpack

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

    # State & Gates
    ## Forget Gate
    ft:np.array = None                  # Forget gate output
    ft_history:np.array = None
    ## Input Gate
    it:np.array = None                  # Input gate output
    it_history:np.array = None
    ctt:np.array = None                 # Candidate
    ctt_history:np.array = None
    # Cell State
    ct:np.array = None                  # New cell state
    ct_history:np.array = None
    # Output Gate
    ot:np.array = None                  # Output gate output
    ot_history:np.array = None
    ht:np.array = None                  # Hidden state
    ht_history:np.array = None
    
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
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
        elif activation.lower() not in ['relu', 'sigmoid', 'tanh']:
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
        Calculate the given input tensor
        """
        # Clear state and gates history
        self.reset_state_and_gates()
        if input.shape != self.input_shape[1:]:
            # Check input shape
            raise ValueError(f"Expected shape={self.input_shape}, \
found shape={input.shape}")
        for timestep in input:
            # Forget Gate
            ft = lstm_fpack['sigmoid'](
                self.U_forget @ timestep.T
                + self.W_forget @ self.ht
                + self.b_forget)
            self.ft = ft
            self.ht_history = np.hstack((self.ht_history, ft))
            
            # Input Gate
            it = lstm_fpack['sigmoid'](
                self.U_input @ timestep.T
                + self.W_input @ self.ht
                + self.b_input)
            self.it = it
            self.it_history = np.hstack((self.it_history, it))
            
            ctt = lstm_fpack[self.algorithm](
                self.U_cell @ timestep.T
                + self.W_cell @ self.ht
                + self.b_cell)
            self.ctt = ctt
            self.ctt_history = np.hstack((self.ctt_history, ctt))
            # Cell State
            ct = ft * self.ct + it * ctt
            self.ct = ct
            self.ct_history = np.hstack((self.ct_history, ct))

            # Output Gate
            ot = lstm_fpack['sigmoid'](
                self.U_output @ timestep.T
                + self.W_output @ self.ht
                + self.b_output)
            self.ot = ot
            self.ot_history = np.hstack((self.ot_history, ot))

            ht = ot * lstm_fpack[self.algorithm](ct)
            self.ht = ht
            self.ht_history = np.hstack((self.ht_history, ht))
        
        if self.return_sequences:
            self.output = self.ht_history[input.shape[:,1:]].T.copy()
        else:
            self.output = self.ht_history[:,-1].copy()
        return self.output
    
    def reset_state_and_gates(self):
        """
        Clear state and gates to reset timestep calculations
        """
        ## Forget Gate
        self.ft = np.zeros((self.num_of_units, 1))
        self.ft_history = self.ft.copy()
        ## Input Gate
        self.it = np.zeros((self.num_of_units, 1))
        self.it_history = self.it.copy()
        self.ctt = np.zeros((self.num_of_units, 1))
        self.ctt_history = self.ctt.copy()
        # Cell State
        self.ct = np.zeros((self.num_of_units, 1))
        self.ct_history = self.ct.copy()
        # Output Gate
        self.ot = np.zeros((self.num_of_units, 1))
        self.ot_history = self.ot.copy()
        self.ht = np.zeros((self.num_of_units, 1))
        self.ht_history = self.ht.copy()

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
        Generate weights matrix from current input_shape
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
        obj['data']['return_sequences'] = self.return_sequences
        obj['data']['num_of_units'] = self.num_of_units
        obj['data']['U_forget'] = self.U_forget
        obj['data']['U_input'] = self.U_input
        obj['data']['U_cell'] = self.U_cell
        obj['data']['U_output'] = self.U_output
        obj['data']['W_forget'] = self.W_forget
        obj['data']['W_input'] = self.W_input
        obj['data']['W_cell'] = self.W_cell
        obj['data']['W_output'] = self.W_output
        obj['data']['b_forget'] = self.b_forget
        obj['data']['b_input'] = self.b_input
        obj['data']['b_cell'] = self.b_cell
        obj['data']['b_output'] = self.b_output

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
        self.return_sequences = bool(object['return_sequences'])
        self.num_of_units = object['num_of_units']
        self.U_forget = np.array(object['U_forget'])
        self.U_input = np.array(object['U_input'])
        self.U_cell = np.array(object['U_cell'])
        self.U_output = np.array(object['U_output'])
        self.W_forget = np.array(object['W_forget'])
        self.W_input = np.array(object['W_input'])
        self.W_cell = np.array(object['W_cell'])
        self.W_output = np.array(object['W_output'])
        self.b_forget = np.array(object['b_forget'])
        self.b_input = np.array(object['b_input'])
        self.b_cell = np.array(object['b_cell'])
        self.b_output = np.array(object['b_output'])
        

if __name__ == "__main__":
    pass