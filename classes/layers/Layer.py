from abc import ABC, abstractmethod, abstractproperty

class Layer(ABC):
    """
    Abstract class for layer
    """
    ## PROPERTIES
    @property
    @abstractmethod
    def name(self):
        """
        Layer name
        """
        pass

    @property
    @abstractmethod
    def algorithm(self):
        """
        Layer algorithm
        """
        pass

    @property
    @abstractmethod
    def input(self):
        """
        Save recent input to class properties
        """
        pass
    
    @property
    @abstractmethod
    def output(self):
        """
        Save recent to class properties
        """
        pass

    @property
    @abstractmethod
    def input_shape(self):
        """
        Details for layer's input shape
        """
        pass
    
    @property
    @abstractmethod
    def output_shape(self):
        """
        Details for layer's output shape
        """
        pass
    

    ## METHODS
    @classmethod
    @abstractmethod
    def calculate(self, input):
        """
        Calculate the given input to the desired output
        Also saves input and output to class properties
        """
        pass
    
    @classmethod
    @abstractmethod
    def update(self):
        """
        Update weights from the calculated error-term
        """
        pass
    
    @classmethod
    @abstractmethod
    def compile(self, input_shape):
        """
        COMPILING PURPOSE
        Compile layer with the given input shape
        """
        pass
    
    @classmethod
    @abstractmethod
    def calculate_output_shape(self):
        """
        COMPILING PURPOSE
        Calculate ouput shape from layer's input shape
        """
        pass
    
    
    @classmethod
    @abstractmethod
    def to_object(self):
        """
        SAVING/LOADING PURPOSE
        Convert self to json-like object (dictionary)
        """
        pass

    @classmethod
    @abstractmethod
    def from_object(self, object):
        """
        SAVING/LOADING PURPOSE
        Convert json-like object (dictionary) to layer object
        """
        pass