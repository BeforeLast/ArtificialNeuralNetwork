from cProfile import label
import keras
import numpy as np
import tensorflow as tf

class ImageConvert():
  dir_path:str = "data/test"

  def __init__(self):
    pass

  def convert(self):
    """
    Returns a 2D array with each elmt consists of image_data and its label
    """

    ds = keras.utils.image_dataset_from_directory(
      self.dir_path, 
      color_mode="rgb",
      label_mode = "int",
      batch_size = None
    )
    
    # normalize
    normalization_layer = keras.layers.Rescaling(1./255)
    normalized_ds = ds.map((lambda x, y: (normalization_layer(x), y)))

    result = []
    for image, label in normalized_ds:
      result.append([image.numpy(), label.numpy()])

    return np.asarray(result)