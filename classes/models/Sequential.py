from typing import Union

from classes.layers.Layer import Layer


class Sequential():
    """
    Sequential ANN model type
    """
    ## PROPERTIES
    # Model info
    name:str = None
    layers:list[Layer] = None

    # Training info
    optimizer:str = None
    loss:str = None
    metrics:Union[list[str], str] = None

    ## METHODS
    def __init__(self):
        """
        Class constructor
        """
        pass

    def compile(self, optimizer, loss, metrics):
        """
        Compile the model by adjusting neural connection and training/test options
        """
        pass

    def fit(self, data, label):
        """
        ! FOR MILESTONE 2 !
        Train the model from the given data and label
        """
        pass

    def predict(self, data):
        """
        Predict the labels from the given data
        """
        pass

    def save(self, dir):
        """
        ! FOR MILESTONE 2 !
        Save model to a file
        """
        pass
    
    def load(self, dir):
        """
        ! FOR MILESTONE 2 !
        Load model from a file
        """
        pass
    
    # Debugging methods
    def summary(self):
        """
        Return the summary (in the form of string) of the model
        """
        pass

    def get_layer(self, name=None, index=None) -> Layer:
        """
        Return the selected selected by name or index,
        if name is given, then index is ignored
        """
        pass