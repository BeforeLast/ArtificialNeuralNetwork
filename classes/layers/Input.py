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
    ouput_shape:Union[int, tuple, list] = None
    
    def __init__(self, ):
        """
        Class constructor
        """
        pass


    def calculate(self, input):
        """
        Return input data as array from image folder
        """

        data = ImageConvert().convert()
        return data

    def update(self):
        """
        ! IGNORE !
        Does not exist for this layer
        """
        pass