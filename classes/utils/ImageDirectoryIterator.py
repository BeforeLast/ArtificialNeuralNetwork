# from classes.utils.ImageConvert import ImageConvert
import numpy as np
from PIL import Image
from random import uniform
from typing import Dict, List, Tuple

from classes.misc.Constants import IMAGEDIRECTORYITERATOR_CONST

class ImageDirectoryIterator():
    """
    ImageDirectoryIterator is a class to save directory based on labels
    """
    image_convert = None

    # LABELLING
    labels_decoding:Dict[int,str] = {}
    # sample:
    # { 0:"cat", 1:"dog" }
    labels_encoding:Dict[str,int] = {}
    # sample:
    # { "cat":0, "dog":1 }
    labels:List[str] = []
    # sample:
    # ["cat", "dog"]
    label_mode:str = None
    color_mode:str = None

    # IMAGE AUGMENTATION
    target_size:Tuple[int,int] = None

    # BATCH ITERATOR
    directory_data:Dict[int,List[str]] = {}
    # sample:
    # {
    #   1: ["./data/cat/1.jpg", "./data/cat/2.jpg"],
    #   2: ["./data/dog/1.jpg", "./data/dog/2.jpg"]
    # }

    # Shuffle batch
    shuffle_data_label:List[dict] = []
    # sample:
    # [
    #   {"data":"./data/dog/1.jpg"", "label":2},
    #   {"data":"./data/cat/1.jpg"", "label":1}
    # ]
    current_idx:int = None
    

    def __init__(self, image_convert,
            label_mode:str, color_mode:str,
            target_size:Tuple[int, int]):
        self.image_convert = image_convert
        self.label_mode = label_mode
        self.color_mode = color_mode
        self.target_size = target_size
        self.directory_data = {}
        self.shuffle_data_label = []
        self.current_idx = None
        self.labels_decoding = {}
        self.labels_encoding = {}

    def shuffle(self):
        # Converting from directory data to list of object consisting of label
        # and data
        self.shuffle_data_label = []
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
            # reshuffle data
            self.shuffle()
            # stop is current_idx is none (reached the end or not started yet)
            return
        # open image 
        image = Image.open(self.shuffle_data_label[self.current_idx]["data"])

        # image_mode
        img_mode = image.mode

        # resize
        image.thumbnail(self.target_size, Image.ANTIALIAS)

        # add zero padding
        padding_color = IMAGEDIRECTORYITERATOR_CONST\
            ['zero_padding'][img_mode.lower()]
        padded_image = Image.new(img_mode,
            self.target_size,
            padding_color)
        center_paste = (
            (self.target_size[0] - image.size[0]) // 2,
            (self.target_size[1] - image.size[1]) // 2)
        padded_image.paste(image, center_paste)

        # rotate
        padded_image = padded_image.rotate(
            uniform(-self.image_convert.rotate, self.image_convert.rotate))

        # get image data as array
        data = np.array(padded_image)

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