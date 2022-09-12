from PIL import Image
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

    max_height, max_width = self.get_largest_dimension()

    result = []

    for root, dirs, files in os.walk("data/test"):
      if (root == "data/test\dogs"):
        for name in files:
          # open image 
          image = Image.open(os.path.join(root,name))

          # resize with zero pad
          resized_image = tf.image.resize_with_pad(image, max_height, max_width)

          # get image data as array
          data = np.array(resized_image)

          # set label to 0 (dogs)
          label = 0

          result.append([data, label])
      
      elif (root == "data/test\cats"):
        for name in files:
          # open image 
          image = Image.open(os.path.join(root,name))

          # resize with zero pad
          resized_image = tf.image.resize_with_pad(image, max_height, max_width)

          # get image data as array
          data = np.array(resized_image)

          # set label to 1 (cats)
          label = 1

          result.append([data, label])

    return np.asarray(result)