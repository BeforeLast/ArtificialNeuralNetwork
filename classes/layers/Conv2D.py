# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

from typing import Union
import classes.layers.Layer
import numpy as np

class Conv2D(classes.layers.Layer.Layer):
    """
    Convolutional layer with 2D input
    """
    # Layer info
    name = None
    input = None
    output = None
    input_shape:Union[int, tuple, list] = None
    ouput_shape:Union[int, tuple, list] = None

    # Convolution info
    padding_size:int = None
    conv_kernel_size:Union[int, tuple, list] = None
    conv_stride:Union[int, tuple, list] = None
    filters:int = None

    # Detector info
    algorithm:str = None

    # Pooling info
    pool_kernel_size:Union[int, tuple, list] = None
    pool_stride:Union[int, tuple, list] = None
    pool_mode:str = None
    
    def __init__(self, ):
        """
        Class constructor
        """
        pass


    def calculate(self, input):
        """
        Calculate the given input tensor to given output
        """
        conv_res = self.convolve(input)
        detc_res = self.detect(conv_res)
        pool_res = self.pool(detc_res)
        pass

    def update(self):
        """
        Update the layers' weight
        """
        pass
    
    def convolve(self, input):
        """
        Convolve the layer
        """
        pass
    
    def detect(self, input):
        """
        Detect the given input using activation function
        """
        pass

    def pool(self, input):
        """
        Pool the given input using the layer pool method
        """

    