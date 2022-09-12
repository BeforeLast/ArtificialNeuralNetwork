from PIL import Image
import keras
import numpy as np
import os
import tensorflow as tf

class ImageConvert():
  dir_path:str = "data/test"

  def __init__(self):
    pass

  def get_largest_dimension(self):
    """
    Returns a tuple consists of max height and max width of all images
    """

    max_height = 0
    max_width = 0

    for root, dirs, files in os.walk(self.dir_path):
      for name in files:
        if name.endswith('.jpg'):
          data = np.array(Image.open(os.path.join(root,name)))
          if (data.shape[0] > max_height) : max_height = data.shape[0]
          if (data.shape[1] > max_width) : max_width = data.shape[1]
    
    return (max_height, max_width)
              

  def convert(self):
    """
    Returns a 2D array with each elmt consists of image_data and its label
    """

    ds = keras.utils.image_dataset_from_directory(
      self.dir_path,
      batch_size = None
    )
    
    # normalize
    normalization_layer = keras.layers.Rescaling(1./255)
    normalized_ds = ds.map((lambda x, y: (normalization_layer(x), y)))

    # get largest dimension
    max_height, max_width = self.get_largest_dimension()

    result = []
    for image, label in normalized_ds:
      # resize with zero padding
      resized_image = tf.image.resize_with_pad(image, max_height, max_width)
      result.append([resized_image.numpy(), label.numpy()])

    return np.asarray(result)