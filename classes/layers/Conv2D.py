# Guide : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

from classes.layers.Layer import Layer as BaseLayer
from classes.misc.Function import conv2d_fpack, misc
from scipy.signal import fftconvolve
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, List

class Conv2D(BaseLayer):
    """
    Convolutional layer with 2D input
    """
    # Layer info
    name:str = None
    input = None
    output:np.ndarray = None
    input_shape:Tuple[None, int, int, int] = None
    output_shape:Tuple[None, int, int, int] = None

    # Convolution info
    num_of_filters:int = None
    conv_kernel_size:Tuple[int, int] = None
    conv_padding_size:int = None
    conv_stride:Tuple[int, int] = None
    conv_filters:List[Tuple[List[np.ndarray], float]] = None
    conv_output_shape:Tuple[None, int, int, int] = None
    conv_output:np.ndarray = None

    # Detector info
    algorithm:str = None
    detector_output:np.ndarray = None

    # Pooling info
    pool_kernel_size:Tuple[int, int] = None
    pool_stride:Tuple[int, int] = None
    
    # Deltas
    deltas_wrt_filters:List[Tuple[List[np.ndarray], float]] = None
    deltas_wrt_inputs:np.ndarray = None
    delta_pools = None
    delta_detectors = None
    
    num_params:int = None
    
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
            self.conv_kernel_size = tuple(conv_kernel_size)
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
            self.conv_stride = tuple(conv_stride)
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
            self.pool_kernel_size = tuple(pool_kernel_size)
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
            self.pool_stride = tuple(pool_stride)
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

    def backward(self, next_layer = None, target = None):
        """
        Update the layer's delta (perform back propagation)
        """ 
        self.delta_pools = []
        self.delta_detectors = []
        self.deltas_wrt_filters = []
        self.deltas_wrt_inputs = [np.zeros(self.input_shape[-3:-1])]
        for i in range(self.num_of_filters):
            # Gradient in pooling
            if self.pool_mode in ["max", "maximum"]:
                delta_pool = np.zeros(self.conv_output_shape[-3:-1])
                for j in range(0, self.conv_output_shape[-3], self.pool_stride[0]):
                    for k in range(0, self.conv_output_shape[-2], self.pool_stride[1]):
                        window = self.conv_output[j:j + self.pool_kernel_size[0], k:k + self.pool_kernel_size[1], i]
                        max_el = np.amax(window)
                        bool_window = (window == max_el).astype(int)
                        delta_pool[j:j + self.pool_kernel_size[0], k:k + self.pool_kernel_size[1]] = bool_window
                self.delta_pools.append(delta_pool)
            elif self.pool_mode in ["avg", "average"]:
                delta_pool = np.full(self.conv_output_shape[-3:-1], np.mean(self.detector_output[i]))
                self.delta_pools.append(delta_pool)
                
            # Gradient in detector (relu)
            delta_detector = (self.conv_output[:,:,i] > 0).astype(int)
            self.delta_detectors.append(delta_detector)            
            
            delta = self.delta_detectors[i].T @ self.delta_pools[i]

            curr_deltas_wrt_inputs = [np.zeros(self.input_shape[-3:-1])]                    
            curr_deltas_wrt_filters = []

            next_layer_delta = None
            if (type(next_layer) is Conv2D) :
                # Next layer is conv
                next_layer_delta = next_layer.deltas_wrt_inputs[i]
            else :
                # Next layer is dense
                next_layer_delta = []

                for weight in next_layer.weights:
                    curr = np.dot(next_layer.deltas_wrt_inputs, weight)
                    next_layer_delta.append(curr)
            
            for channel_idx in range(self.input_shape[-1]):
                if (type(next_layer) is Conv2D):
                    # Next layer is conv
                    curr_deltas_wrt_output = delta[channel_idx].T @ next_layer_delta
                    curr_deltas_wrt_filter = fftconvolve(
                        input[:,:,j], curr_deltas_wrt_output, mode='valid'
                    )
                    curr_deltas_wrt_inputs = fftconvolve(
                        np.rot90(self.conv_filters[i][0][channel_idx], 2), curr_deltas_wrt_output, mode='full'
                    )
                    curr_deltas_wrt_filters.append(curr_deltas_wrt_filter)

                    self.deltas_wrt_inputs[channel_idx] = self.deltas_wrt_inputs[channel_idx] + curr_deltas_wrt_inputs
                else :
                    # Next layer is dense
                    delta_cross_input = []
                    for idx, delta in enumerate(next_layer_delta):
                        curr = [delta * next_layer.output[idx-1]]
                        delta_cross_input.append(curr)

                    curr_deltas_wrt_filters.append(delta_cross_input)

            self.deltas_wrt_filters.append(curr_deltas_wrt_filters)

        self.deltas_wrt_filters = np.asarray(self.deltas_wrt_filters)

    def update(self, learning_rate):
        """
        Update the layers' weight
        """
        # Update filter weight
        for i in range(self.num_of_filters):
            for channel_idx in range(self.input_shape[-1]):
                delta_weight = self.deltas_wrt_filters[i][channel_idx] * learning_rate
                self.conv_filters[i][0][channel_idx] = self.conv_filters[i][0][channel_idx] - delta_weight            
    
    # DONE : FIX CONVOLVE
    def convolve(self, input):
        """
        Convolve the layer
        """
        output = None
        # Copy input data to prevent reference edit
        data_copy:np.ndarray = input.copy()
        # Iterate through filters
        for filters, bias in self.conv_filters:
            # Initiate single filter output 
            single_filter_output = np.zeros(self.conv_output_shape[-3:-1])
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
                temp_single_output = fftconvolve(
                    temp_channel_data, filters[channel], mode='valid')
                # Stride process
                temp_single_output = temp_single_output[
                    ::self.conv_stride[0],
                    ::self.conv_stride[1]]
                # Sum channel output to single output
                single_filter_output += temp_single_output
            # Add bias to sum of channel(s) output
            single_filter_output += bias
            # Add to output channel
            if output is None:
                # Instantiate output data
                output = single_filter_output
            else:
                # Stack channel
                output = np.dstack((output, single_filter_output))
        self.conv_output = output.copy()
        return output
    
    # DONE : FIX DETECTOR
    def detect(self, input):
        """
        Detect the given input using activation function
        """
        # Detetor process
        detector_result = []
        detector_result = conv2d_fpack['relu'](input)
        self.detector_output = detector_result.copy()
        return detector_result
        
    # DONE : FIX POOLING
    def pool(self, input):
        """
        Pool the given input using the layer pool method
        """
        pool_input_shape = input.shape[-3:]
        
        ## DEBUGGING PURPOSE ##
        # See expected pool output shape
        pool_output_shape = (
            misc['expected_output_dim_length'](
                pool_input_shape[0], self.pool_kernel_size[0],
                0, self.pool_stride[0]),
            misc['expected_output_dim_length'](
                pool_input_shape[1], self.pool_kernel_size[1],
                0, self.pool_stride[1]),
            pool_input_shape[-1]
        )
        ##

        # Setting pooling mode
        pool_mode_function = None
        if self.pool_mode in ["max", "maximum"]:
            pool_mode_function = np.max
        elif self.pool_mode in ["avg", "average"]:
            pool_mode_function = np.average
        
        # Pooling process
        pool_result = None
        for channel in range(input.shape[-1]):
            channel_data = input[:,:,channel]
            temp_result = []
            # Separate receptive field
            for row in sliding_window_view(channel_data, self.pool_kernel_size):
                temp_row = []
                for column in row:
                    temp_row.append(pool_mode_function(column))
                temp_result.append(temp_row)
            # Convert result to np array
            temp_result = np.array(temp_result)
            # Apply stride
            temp_result[::self.pool_stride[0],::self.pool_stride[1]]
            # Stack result
            if pool_result is None:
                # Instantiate pool result with first channel pool
                pool_result = temp_result
            else:
                # Stack another channel pooling to previous stack
                pool_result = np.dstack((pool_result, temp_result))
        return pool_result
    
    # ANCHOR : COMPILING
    # DONE : FIX COMPILING
    def compile(self, input_shape):
        """
        COMPILING PURPOSE
        Compile layer to be used by calucating output shape and
        instantiating kernels
        """
        if len(input_shape) == 3:
            # Only state data dimension and channel
            fix_shape = [None]
            fix_shape.extend(list(input_shape))
            self.input_shape = tuple(fix_shape)
        else:
            # Batch input shape already stated (None)
            self.input_shape = input_shape
        self.generate_filters()
        self.calculate_output_shape()
        # Set number of params
        self.num_params = self.conv_output_shape[-1] * (
            self.input_shape[-1] * self.conv_kernel_size[0] * self.conv_kernel_size[1] + 1
        )

    # DONE : FIX GENERATE FILTERS
    def generate_filters(self):
        """
        COMPILING PURPOSE
        Generate filters from current input_shape
        """
        # Only generate if not generated yet
        if self.conv_filters is None:
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
    
    # DONE : FIX OUTPUT SHAPE
    def calculate_output_shape(self):
        """
        COMPILING PURPOSE
        Calculate ouput shape from layer's input shape
        """
        # Get input shape
        input_shape = self.input_shape
        output_batch = None
        # Convolution output shape
        pool_y_dim = misc['expected_output_dim_length'](
            input_shape[-3], self.conv_kernel_size[0],
            self.conv_padding_size, self.conv_stride[0])
        pool_x_dim = misc['expected_output_dim_length'](
            input_shape[-2], self.conv_kernel_size[1],
            self.conv_padding_size, self.conv_stride[1])
        self.conv_output_shape = (pool_y_dim, pool_x_dim, self.num_of_filters)
        # Pooling output shape
        output_y_dim = misc['expected_output_dim_length'](
            pool_y_dim, self.pool_kernel_size[0],
            0, self.pool_stride[0])
        output_x_dim = misc['expected_output_dim_length'](
            pool_x_dim, self.pool_kernel_size[1],
            0, self.pool_stride[1])
        # Channel
        # note: output channel will follow num_of_filters
        output_channel = self.num_of_filters
        self.output_shape = (output_batch, output_y_dim, output_x_dim, output_channel)
        return self.output_shape

    def to_object(self):
        """
        SAVING/LOADING PURPOSE
        Convert self to json-like object (dictionary)
        """
        obj = {}
        obj['layer_type'] = 'conv2d'
        obj['data'] = {}
        # Layer data
        obj['data']['name'] = self.name
        obj['data']['input_shape'] = self.input_shape
        obj['data']['output_shape'] = self.output_shape

        # Convolution data
        obj['data']['num_of_filters'] = self.num_of_filters
        obj['data']['conv_kernel_size'] = self.conv_kernel_size
        obj['data']['conv_padding_size'] = self.conv_padding_size
        obj['data']['conv_stride'] = self.conv_stride
        temp_conv_filter = []
        for kernels_bias_pair in self.conv_filters:
            temp_conv_filter.append((
                [kernel.tolist() for kernel in kernels_bias_pair[0]],
                kernels_bias_pair[1]
            ))
        obj['data']['conv_filters'] = temp_conv_filter
        obj['data']['conv_output_shape'] = self.conv_output_shape

        # Detector data
        obj['data']['algorithm'] = self.algorithm

        # Pooling data
        obj['data']['pool_kernel_size'] = self.pool_kernel_size
        obj['data']['pool_stride'] = self.pool_stride
        obj['data']['pool_mode'] = self.pool_mode
        return obj
        

    def from_object(self, object):
        """
        SAVING/LOADING PURPOSE
        Convert json-like object (dictionary) to layer object
        """
        # Layer data
        self.name = object['name']
        self.input_shape = tuple(object['input_shape']) \
            if object['input_shape'] else None
        self.output_shape = tuple(object['output_shape']) \
            if object['output_shape'] else None

        # Convolution data
        self.num_of_filters = object['num_of_filters']
        self.conv_kernel_size = tuple(object['conv_kernel_size'])
        self.conv_padding_size = object['conv_padding_size']
        self.conv_stride = tuple(object['conv_stride'])
        self.conv_filters = []
        for kernels_bias_pair in object['conv_filters']:
            self.conv_filters.append((
                [np.array(kernel) for kernel in kernels_bias_pair[0]],
                kernels_bias_pair[1]
            ))
        self.conv_output_shape = tuple(object['conv_output_shape']) \
            if object['conv_output_shape'] else None

        # Detector data
        self.algorithm = object['algorithm']

        # Pooling data
        self.pool_kernel_size = tuple(object['pool_kernel_size'])
        self.pool_stride = tuple(object['pool_stride'])
        self.pool_mode = object['pool_mode']
    


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
    ## Compile
    c2d_test = Conv2D(
        2, (2,2),
        conv_stride=1, conv_padding_size=0,
        activation='relu',
        pool_kernel_size=(2,2), pool_stride=1,
        pool_mode='max')
    print("TYPE TEST:", type(c2d_test) is Conv2D)
    c2d_test.compile((None, 4, 4, 3))
    print("input shape test:", c2d_test.input_shape == (None, 4, 4, 3))
    print("output shape test:", c2d_test.output_shape == (None, 2, 2, 2))
    print()
    ## Calculate
    c2d_test.calculate(np.random.rand(4, 4, 3))
    print(c2d_test.input)
    print(c2d_test.input.shape)
    print()
    print()
    print(c2d_test.conv_output)
    print(c2d_test.conv_output.shape)
    print()
    print()
    print(c2d_test.output)
    print(c2d_test.output.shape)
    ## Detector
    test_detector = Conv2D(1, (2,2))
    test_detector_array = np.array([[[2, 7.2341], [-6,-0.1226]],
                [[0,-0.1763],[-1,12.316872]]])
    print(test_detector_array)
    print()
    print()
    test_detector_result = test_detector.detect(test_detector_array)
    print(test_detector_result)
    print("TEST SHAPE :", test_detector_array.shape == test_detector_result.shape)
    ## Pooling
    test_pool_1 = Conv2D(
        1, (2,2),
        pool_kernel_size=(2,2),
        pool_stride=1,
        pool_mode='max',
    )
    test_pool_array = np.array([[[2, 7.2341], [-6, -0.1226]],
                [[0, -0.1763],[-1, 12.316872]]])
    print(test_pool_array)
    print(test_pool_array.shape)
    test_pool_result = test_pool_1.pool(test_pool_array)
    print(test_pool_result)
    print(test_pool_result.shape)
