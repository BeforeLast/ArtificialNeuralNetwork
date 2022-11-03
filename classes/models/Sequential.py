import json
from typing import Type, Union
from tabulate import tabulate
from classes.layers.Dense import Dense

from classes.layers.Layer import Layer
import numpy as np
from typing import List, Dict
from classes.misc.Class import layerinit_cpack, lossinit_cpack
from classes.utils.ImageDirectoryIterator import ImageDirectoryIterator
from classes.losses.Losses import Losses

class Sequential():
    """
    Sequential ANN model type
    """
    ## PROPERTIES
    # Model info
    name:str = None
    layers:List[Layer] = None
    input_shape:tuple = None
    output_shape:tuple = None
    compiled:bool = False

    # Training info
    optimizer:str = None
    loss:Losses = None
    metrics:Union[List[str], str] = None

    # Prediction info
    predict_history:Dict[str, any] = None

    ## METHODS
    def __init__(self, layers:List[Layer], **kwargs):
        """
        Class constructor
        """
        self.name = kwargs.get("name", "Sequential")
        self.layers = layers
        self.predict_history = {}
        self.optimizer = None
        self.loss = None
        self.metrics = None

    def compile(self, optimizer='sgd', loss='binary_crossentropy', metrics='accuracy'):
        """
        Compile the model by adjusting neural connection and training/test options
        ! FOR MILESTONE 2 !
        - optimizer
        - loss
        - metrics
        """
        # # Check whether model has layer or not
        if not self.layers or not len(self.layers):
            raise AttributeError("Model does not have any layer")
        # Instantiate ouput shape chain
        output_shape_chain = None

        # Iterate and compile layers sequentially
        for layer in self.layers:
            layer.compile(output_shape_chain)
            output_shape_chain = layer.output_shape
        self.input_shape = self.layers[0].input_shape
        self.output_shape = self.layers[-1].input_shape

        # Instantiate Loss
        if isinstance(loss, Losses):
            self.loss = loss
        else:
            self.loss = lossinit_cpack[loss.lower()]()

        # Save compiled state
        self.compiled = True

    def fit(self, data, label, batch_size, epochs, learning_rate, verbose = True):
        """
        ! FOR MILESTONE 2 !
        Train the model from the given data and label
        """
        for i in range(epochs):
            print(f"Epochs {i+1}/{epochs}")
            batch_index = 0
            print(f"Step: 0/{len(data)}",end='\r')
            batch_iter = 1
            batch_steps = len(data)//batch_size
            while batch_index < len(data):
                batch_input = data[batch_index:min(batch_index + batch_size, len(data))]
                batch_label = label[batch_index:min(batch_index + batch_size, len(label))]

                for j in range(len(batch_input)):
                    curr_input = batch_input[j]
                    curr_label = batch_label[j]
                    
                    # Forward
                    for layer in self.layers:
                        output = layer.calculate(curr_input)
                        curr_input = output

                    # Backward
                    for k in range(len(self.layers)-1, -1, -1):
                        if (type(self.layers[k]) is Dense):
                            # Dense Layer backward
                            if (k < len(self.layers) - 1):
                                # Hidden layer
                                self.layers[k].backward(self.layers[k+1], None)
                            else :
                                # Output layer
                                self.layers[k].backward(None, curr_label)

                # Update Weight
                for j in range(len(self.layers)-1, -1, -1):
                    if (type(self.layers[k]) is Dense):
                        self.layers[j].update(learning_rate)
                
                batch_index += batch_size
                batch_iter += 1
                # Print process
                print(f"Step: {batch_iter}/{batch_steps}",
                    end='\n' if batch_index == batch_steps else '\r')


    def predict(self, data, target=None):
        """
        Predict the labels from the given data
        """
        if type(data) is ImageDirectoryIterator:
            results = []
            true_labels = []
            step = next(data)
            data_len = len(data)
            counter = 0
            while step:
                # Get data from iterator
                result = step['data']
                # Get label from iterator
                true_labels.append(step['label'])
                # Calculate through model
                for layer in self.layers:
                    result = layer.calculate(result)
                results.append(result)
                step = next(data)
                counter += 1
                # Print step
                print(f"Step: {counter}/{data_len}", end=' | ' if counter==data_len else '\r')
            # Reshape predictions and targets shape
            results = np.array(results).reshape(-1,1)
            true_labels = np.array(true_labels).reshape(-1,1)
            # Calculate loss
            self.predict_history['loss'] = self.loss.calculate(
                true_labels, results)[0]
            # Print loss
            print(f"loss: {self.predict_history['loss']}")
            return {
                'results':results,
                'true_labels':true_labels
            }
        elif type(data) in [np.ndarray, list, tuple]:
            # Check data processing type (batch or not batch)
            data_check = np.array(data)
            if data_check.shape == self.input_shape[1:]:
                # Single prediction
                # data shape is the same as input shape
                # meaning it is a single prediction
                result = data
                for layer in self.layers:
                    result = layer.calculate(result)
                return result
            else:
                # Batch prediction
                results = []
                # Iterate step
                for step in data_check:
                    result = step
                    for layer in self.layers:
                        result = layer.calculate(result)
                    results.append(result)
                return results

    def save(self, dir:str):
        """
        Save model to a file
        """
        # Convert model to json-like object
        obj = {}
        obj['name'] = self.name
        obj['input_shape'] = self.input_shape
        obj['output_shape'] = self.input_shape

        # Layer
        layer_obj = []
        for layer in self.layers:
            layer_obj.append(layer.to_object())
        obj['layers'] = layer_obj

        # Add json file extension
        fixed_dir = dir
        if not fixed_dir.endswith('.json'):
            fixed_dir = dir + '.json'
        
        # Save to file
        with open(fixed_dir, 'w') as f:
            json.dump(obj, f, indent=2)
    
    def load(self, dir:str):
        """
        Load model from a file
        """
        with open(dir,'r') as f:
            # Load file
            data = json.load(f)
            # Parse general information
            self.name = data['name']
            self.input_shape = data['input_shape']
            self.output_shape = data['output_shape']
            # Parse layer
            self.layers = []
            for layer in data['layers']:
                temp_layer = layerinit_cpack[layer['layer_type']]()
                temp_layer.from_object(layer['data'])
                self.layers.append(temp_layer)

    
    # Debugging methods
    def summary(self):
        """
        Return the summary (in the form of string) of the model
        """
        layer_infos = []
        sum_params = 0
        for layer in self.layers:
            sum_params += layer.num_params
            layer_infos.append(
                [
                    layer.name,
                    type(layer).__name__,
                    layer.output_shape,
                    layer.num_params
                ]
            )
        print(tabulate(layer_infos, headers=['Name', 'Type', 'Output Shape', 'Num of Params'], tablefmt="mixed_outline"))
        print("Total params: ", sum_params)
        print("Trainable params: ", sum_params)
        print("Non-trainable params: ", 0)

    def get_layer(self, name=None, index=None) -> Layer:
        """
        Return the selected selected by name or index,
        if name is given, then index is ignored
        """
        if name:
            for layer in self.layers:
                if layer.name == name:
                    return layer
        else:
            return self.layers[index]