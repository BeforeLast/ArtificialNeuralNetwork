# from classes.utils.ImageConvert import ImageConvert
import numpy as np
from PIL import Image
import tensorflow as tf
from random import uniform

class ImageDirectoryIterator():
    """
    ImageDirectoryIterator is a class to save directory based on labels
    """
    image_convert = None

    # LABELLING
    labels_decoding:dict[int,str] = {}
    # sample:
    # { 0:"cat", 1:"dog" }
    labels_encoding:dict[str,int] = {}
    # sample:
    # { "cat":0, "dog":1 }
    labels:list[str] = []
    # sample:
    # ["cat", "dog"]
    label_mode:str = None

    # IMAGE AUGMENTATION
    target_size:tuple[int,int] = None

    # BATCH ITERATOR
    directory_data:dict[int,list[str]] = {}
    # sample:
    # {
    #   1: ["./data/cat/1.jpg", "./data/cat/2.jpg"],
    #   2: ["./data/dog/1.jpg", "./data/dog/2.jpg"]
    # }

    # Shuffle batch
    shuffle_data_label:list[dict] = []
    # sample:
    # [
    #   {"data":"./data/dog/1.jpg"", "label":2},
    #   {"data":"./data/cat/1.jpg"", "label":1}
    # ]
    current_idx:int = None
    

    def __init__(self, image_convert,
            label_mode:str, color_mode:str,
            target_size:tuple[int, int]):
        self.image_convert = image_convert
        self.label_mode = label_mode
        self.color_mode = color_mode
        self.target_size = target_size

    def shuffle(self):
        # Converting from directory data to list of object consisting of label
        # and data
        for label in self.directory_data:
            for data in self.directory_data[label]:
                self.shuffle_data_label.append(
                    {
                        "label":label,
                        "data":data
                    }
                )
        # Convert to np ndarray
        self.shuffle_data_label = self.shuffle_data_label
        # Shuffle result
        np.random.shuffle(self.shuffle_data_label)
        self.current_idx = 0
    
    def __next__(self):
        if self.current_idx is None:
            # stop is current_idx is none (reached the end or not started yet)
            return
        # open image 
        image = Image.open(self.shuffle_data_label[self.current_idx]["data"])

        # resize with zero pad
        # resized_image = tf.image.resize_with_pad(image, *self.target_size)

        # resize
        image = image.resize(self.target_size)

        # rotate
        image = image.rotate(
            uniform(-self.image_convert.rotate, self.image_convert.rotate))
        
        # get image data as array
        data = np.array(image)

        # rescale
        data = data * self.image_convert.rescale

        
        # convert image data to object
        result = {
            "data":data,
            "label":self.shuffle_data_label[self.current_idx]["label"]
        }

        # increment iterator
        self.current_idx += 1
        if self.current_idx >= len(self.shuffle_data_label):
            # stop iteration if it reached the end
            self.current_idx = None

        return result
    
    def __len__(self):
        """
        Return the number of data
        """
        return len(self.shuffle_data_label)