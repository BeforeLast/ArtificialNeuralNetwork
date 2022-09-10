# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

from multiprocessing.sharedctypes import Value
from typing import Union
from classes.layers.Layer import Layer as BaseLayer
import numpy as np

class Conv2D(BaseLayer):
    """
    Convolutional layer with 2D input
    """
    # Layer info
    name:str = None
    input = None
    output = None
    input_shape:Union[int, tuple, list] = None
    output_shape:Union[int, tuple, list] = None

    # Convolution info
    filters:int = None
    conv_kernel_size:Union[int, tuple, list] = None
    conv_padding_size:int = None
    conv_stride:Union[int, tuple, list] = None
    conv_kernels:list[np.ndarray] = None

    # Detector info
    algorithm:str = None

    # Pooling info
    pool_kernel_size:Union[int, tuple, list] = None
    pool_stride:Union[int, tuple, list] = None
    pool_mode:str = None
    pool_kernel:np.ndarray = None
    
    def __init__(self, 
            filters, conv_kernel_size,              # Conv configuration
            conv_stride=(1, 1), conv_padding_size=1,# Conv configuration
            activation='relu',                      # Detector configuration
            pool_kernel_size=(2, 2),                # Pool configuration
            pool_stride=(1, 1), pool_mode='max',    # Pool configuration
            **kwargs                                # Misc configuration
            ):
        """
        Class constructor
        """
        # CONFIGURATION
        ## Convolution configuration
        ### Filters
        if type(filters) is not int \
                or filters < 1:
            raise ValueError("Filters value must be an integer equal or \
greater than one")
        self.filters = filters
        ### Kernel size
        if type(conv_kernel_size) is int:
            if conv_kernel_size < 1:
                raise ValueError("Kernel size value must be equal or greater \
than one")
            self.conv_kernel_size = (conv_kernel_size, conv_kernel_size)
        elif (type(conv_kernel_size) is tuple           \
                or type(conv_kernel_size) is list)      \
                and len(conv_kernel_size) == 2          \
                and type(conv_kernel_size[0]) is int    \
                and type(conv_kernel_size[1]) is int:
            if conv_kernel_size[0] < 1 or conv_kernel_size[1] < 1:
                raise ValueError("Kernel size value must be equal or greater \
than one")
            self.conv_kernel_size = conv_kernel_size
        else:
            raise ValueError("Kernel size must be a single integer or a \
tuple/list of two integers")
        ### Stride
        if type(conv_stride) is int:
            if conv_stride < 1:
                raise ValueError("Stride value must be an integer equal or \
greater than one")
            self.conv_stride = (conv_stride, conv_stride)
        elif (type(conv_stride) is tuple           \
                or type(conv_stride) is list)      \
                and len(conv_stride) == 2          \
                and type(conv_stride[0]) is int    \
                and type(conv_stride[1]) is int:
            if conv_stride[0] < 1 or conv_stride[1] < 1:
                raise ValueError("Stride value must be equal or greater than \
one")
            self.conv_stride = conv_stride
        else:
            raise ValueError("Stride must be a single integer or a tuple/list \
of two integers")
        ### Padding
        if type(conv_padding_size) is not int \
                or conv_padding_size < 0:
            raise ValueError("Padding value must be an integer equal or \
greater than zero")
        self.conv_padding_size = conv_padding_size

        ## Detector configuration
        ### Activation function
        if type(activation) is not str:
            raise ValueError("Invalid activation parameter type")
        elif activation.lower() not in ["relu"]:
            raise ValueError("Activation function not supported")
        self.algorithm = activation

        ## Pool configuration
        ### Pool kernel size
        if type(pool_kernel_size) is int:
            if pool_kernel_size < 1:
                raise ValueError("Kernel size value must be equal or greater \
than one")
            self.pool_kernel_size = (pool_kernel_size, pool_kernel_size)
        elif (type(pool_kernel_size) is tuple           \
                or type(pool_kernel_size) is list)      \
                and len(pool_kernel_size) == 2          \
                and type(pool_kernel_size[0]) is int    \
                and type(pool_kernel_size[1]) is int:
            if pool_kernel_size[0] < 1 or pool_kernel_size[1] < 1:
                raise ValueError("Kernel size value must be equal or greater \
than one")
            self.pool_kernel_size = pool_kernel_size
        else:
            raise ValueError("Kernel size must be a single integer or a \
tuple/list of two integers")
        ### Pool stride size
        if type(pool_stride) is int:
            if pool_stride < 1:
                raise ValueError("Stride value must be an integer equal or \
greater than one")
            self.conv_stride = (pool_stride, pool_stride)
        elif (type(pool_stride) is tuple           \
                or type(pool_stride) is list)      \
                and len(pool_stride) == 2          \
                and type(pool_stride[0]) is int    \
                and type(pool_stride[1]) is int:
            if pool_stride[0] < 1 or pool_stride[1] < 1:
                raise ValueError("Stride value must be equal or greater than \
one")
            self.pool_stride = pool_stride
        else:
            raise ValueError("Stride must be a single integer or a tuple/list \
of two integers")
        ### Pooling mode
        if type(pool_mode) is not str:
            raise ValueError("Invalid activation parameter type")
        elif pool_mode.lower() not in ["max", "maximum", "average", "avg"]:
            raise ValueError("Activation function not supported")
        
        ## Misc configuration
        self.name = kwargs.get("name", "Conv2D")


        # KERNELS INITIATION
        self.conv_kernels = []
        ## Convolution kernels
        for _ in range(self.filters):
            self.conv_kernels.append(
                np.random.rand(*self.conv_kernel_size)
            )
        ## Pooling kernel
        self.pool_kernel = np.ones(self.pool_kernel_size)


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
        pass

if __name__ == "__main__":
    # TEST
    ## Constructor
    c2d_layer_1 = Conv2D(1, (2, 2))
    c2d_layer_2 = Conv2D(
        2, (4, 4), conv_stride=1, conv_padding_size=2,
        activation='relu',
        pool_kernel_size=9,
        pool_stride=2,
        pool_mode='avg',
        name='c2d_l2')
    ## Class preview
    ### Conv kernels
    print(c2d_layer_1.conv_kernels)
    print(c2d_layer_2.conv_kernels)
    ### Pool kernel
    print(c2d_layer_1.pool_kernel)
    print(c2d_layer_2.pool_kernel)