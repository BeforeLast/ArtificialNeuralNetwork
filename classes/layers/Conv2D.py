# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

from tkinter import E
from typing import Union
from classes.layers.Layer import Layer as BaseLayer
from classes.misc.Function import conv2d_fpack, misc
from scipy.signal import convolve
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
    num_of_filters:int = None
    conv_kernel_size:Union[int, tuple, list] = None
    conv_padding_size:int = None
    conv_stride:Union[int, tuple, list] = None
    conv_filters:tuple[list[np.ndarray], float] = None
    conv_output_shape:tuple[int, int, int, int] = None

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
        self.num_of_filters = filters
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
            self.pool_stride = (pool_stride, pool_stride)
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
        self.pool_mode = pool_mode.lower()
        
        ## Misc configuration
        self.name = kwargs.get("name", "Conv2D")


        # KERNELS INITIATION
        self.conv_filters = []
        ## Convolution kernels
        for _ in range(self.num_of_filters):
            self.conv_filters.append(
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
        self.input = input.copy()
        self.output = pool_res.copy()
        return pool_res

    def update(self):
        """
        Update the layers' weight
        """
        pass
    
    def convolve(self, input):
        """
        Convolve the layer
        """
        output = []
        # Iterate batch
        for data in input:
            # Copy data to prevent reference edit
            data_copy:np.ndarray = data.copy()
            # Iterate through filters
            for filters, bias in self.conv_filters:
                # Initiate single filter output 
                single_output = np.zeros(self.conv_output_shape)
                # Reshape data to match channel
                data_copy = data_copy.reshape(self.input_shape[1:])
                # Iterate through channel
                for channel in range(self.input_shape[-1]):
                    # Split channel data
                    temp_channel_data = data_copy[:,:,channel]
                    # Add padding
                    temp_channel_data = np.pad(
                        temp_channel_data, self.conv_padding_size,
                        mode='constant', constant_values=0)
                    # Convolve process
                    temp_single_output = []
                    for y in range(0,
                        temp_channel_data.shape[0], self.conv_stride[0]):
                        temp_row = []
                        for x in range(0,
                            temp_channel_data.shape[1], self.conv_stride[1]):
                            # Get receptive field
                            receptive_field = temp_channel_data[
                                y:y+self.conv_kernel_size[0],
                                x:x+self.conv_kernel_size[1]]
                            if receptive_field.shape != self.conv_kernel_size:
                                # Skip if receptive field is not the same
                                # shape as kernel
                                continue
                            else:
                                # Convolve and sum
                                temp_row.append(
                                    np.sum(
                                        convolve(
                                        receptive_field,
                                        filters[channel],
                                        mode='same')))
                        if len(temp_row):
                            # Add row to single channel output
                            temp_single_output.append(temp_row)
                        else:
                            # Stop stride convolution if row empty
                            break
                    # Convert channel convolve to np ndarray
                    temp_single_output = np.array(temp_single_output)
                    # Sum channel output to single output
                    single_output += temp_single_output
                # Add bias to summ of channel(s) output
                single_output += bias
                # Add to batch output
                output.append(single_output)
        return output
    
    def detect(self, input):
        """
        Detect the given input using activation function
        """
        # Detetor process
        detector_result = []
        for feature_map in input:
            detector_result.append(conv2d_fpack['relu'](feature_map))
        return detector_result
        

    def pool(self, input):
        """
        Pool the given input using the layer pool method
        """
        
        ## DEBUGGING PURPOSE ##
        # See expected pool output shape
        pool_input_shape = input[0].shape
        pool_output_shape = (
            misc['expected_output_dim_length'](
                pool_input_shape[0], self.pool_kernel_size[0],
                0, self.pool_stride[0]),
            misc['expected_output_dim_length'](
                pool_input_shape[1], self.pool_kernel_size[1],
                0, self.pool_stride[1]) 
        )
        ##

        # Setting pooling mode
        pool_mode_function = None
        if self.pool_mode in ["max", "maximum"]:
            pool_mode_function = np.max
        elif self.pool_mode in ["avg", "average"]:
            pool_mode_function = np.average
        
        # Pooling process
        pool_result = []
        for detector in input:
            temp = []
            for y in range(0, len(detector), self.pool_stride[0]):
                temp_row = []
                for x in range(0, len(detector[0]), self.pool_stride[1]):
                    receptive_field = detector[
                        y:y+self.pool_kernel_size[0],
                        x:x+self.pool_kernel_size[1]]
                    if receptive_field.shape != self.pool_kernel_size:
                        # continue if receptive field does not have shape of
                        # pool's kernel size
                        continue
                    temp_row.append(pool_mode_function(receptive_field))
                if len(temp_row) > 0:
                    # skip pool result for the row if the row is empty due to
                    # the difference in size of receptive field and kernel
                    temp.append(temp_row)
            pool_result.append(np.array(temp))
        return pool_result
    
    def compile(self, input_shape):
        """
        COMPILING PURPOSE
        Compile layer to be used by calucating output shape and
        instantiating kernels
        """
        self.input_shape = input_shape
        self.generate_filters()
        self.calculate_output_shape()

    def generate_filters(self):
        """
        COMPILING PURPOSE
        Generate filters from current input_shape
        """
        # Get number of input channel(s)
        num_of_channels = self.input_shape[-1]
        self.conv_filters = []
        for filter in range(self.num_of_filters):
            # Generate _ filter(s)
            temp_filters = []
            for channel in range(num_of_channels):
                # Generate channel filter(s)
                temp_filters.append(np.random.rand(*self.conv_kernel_size))
            temp_bias = np.random.rand(1)[0]
            self.conv_filters.append((temp_filters, temp_bias))
    
    def calculate_output_shape(self, input_shape=None):
        """
        Calculate ouput shape from layer's input shape or the given
        input shape
        """
        # Get input shape
        if not input_shape:
            input_shape = self.input_shape
        output_batch = self.num_of_filters * input_shape[0]
        # Convolution output shape
        pool_y_dim = misc['expected_output_dim_length'](
            input_shape[1], self.conv_kernel_size[0],
            self.conv_padding_size, self.conv_stride[0])
        pool_x_dim = misc['expected_output_dim_length'](
            input_shape[2], self.conv_kernel_size[1],
            self.conv_padding_size, self.conv_stride[1])
        self.conv_output_shape = (pool_y_dim, pool_x_dim)
        # Pooling output shape
        output_y_dim = misc['expected_output_dim_length'](
            pool_y_dim, self.pool_kernel_size[0],
            0, self.pool_stride[0])
        output_x_dim = misc['expected_output_dim_length'](
            pool_x_dim, self.pool_kernel_size[1],
            0, self.pool_stride[1])
        # Channel
        # note: only 1 channel will be resulted from convolging regardless
        #       the number of input channel(s)
        output_channel = 1
        self.output_shape = (output_batch, output_y_dim, output_x_dim, output_channel)
        return self.output_shape


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
    print(c2d_layer_1.conv_filters)
    print(c2d_layer_2.conv_filters)
    ### Pool kernel
    print(c2d_layer_1.pool_kernel)
    print(c2d_layer_2.pool_kernel)
    ## Compile
    c2d_test_compile = Conv2D(
        2, (3,3),
        conv_stride=1, conv_padding_size=0,
        activation='relu',
        pool_kernel_size=(2,2), pool_stride=1,
        pool_mode='max')
    c2d_test_compile.compile((10, 28, 28, 3))
    print("input shape test:", c2d_test_compile.input_shape == (10, 28, 28, 3))
    print("output shape test:", c2d_test_compile.output_shape == (20, 25, 25, 1))
    ## Calculate
    c2d_test_compile.calculate(np.random.rand(10, 28, 28, 3))
    print(c2d_test_compile.input[0].shape)
    print(c2d_test_compile.output[0].shape)
    ## Detector
    test_detector = Conv2D(1, (2,2))
    test_detector_array = []
    test_detector_array.append(np.array([[2,-6],[0,-1]]))
    test_detector_array.append(np.array([[7.2341,-0.1226],[-0.1763,12.316872]]))
    print(test_detector_array)
    print()
    print(test_detector.detect(test_detector_array))
    ## Pooling
    test_pool_1 = Conv2D(
        1, (2,2),
        pool_kernel_size=(2,2),
        pool_stride=(1,1),
        pool_mode='max',
    )
    test_pool_2 = Conv2D(
        1, (2,2),
        pool_kernel_size=3,
        pool_stride=1,
        pool_mode='avg',
    )
    test_pool_array = []
    test_pool_array.append(np.array([[2,1,2],[0,2,-1],[3,1,5]]))
    test_pool_array.append(np.array([[7.23,-0.12,12.53],[2,1.25,3.63],[-0.17,12.32,2.22]]))
    print(test_pool_1.pool(test_pool_array))
    print(test_pool_2.pool(test_pool_array))
