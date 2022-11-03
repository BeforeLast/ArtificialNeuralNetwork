import os

from classes.utils.ImageDirectoryIterator import ImageDirectoryIterator
from typing import Tuple

class ImageConvert():
  rotate:float = None
  rescale:float = None

  def __init__(self, rotate:float, rescale:float):
    self.rotate = rotate
    self.rescale = rescale

  def from_directory(self, directory:str, target_size:Tuple[int,int],
    mode:str='binary', color_mode:str='rgb'):
    """
    Returns a 2D array with each elmt consists of image_data and its label
    """
    # Check target size
    if type(target_size) not in [Tuple, List]:
      raise TypeError("Invalid type for target size")
    elif len(target_size) != 2:
      raise ValueError("Invalid target size format")
    else:
      target_size = Tuple(target_size)
    # Check color mode
    if color_mode.lower() not in ['rgb','grayscale','rgba']:
      raise NotImplementedError("Unsuported color mode, please use \
'grayscale', 'rgb', or 'rgba' for color mode")
    # Check mode
    if mode.lower() not in ['categorical','binary']:
      raise NotImplementedError("Unsuported classification mode, please use \
'binary' or 'categorical' for classification mode")

    # Instantiate ImageDirectoryIterator
    image_di = ImageDirectoryIterator(self,
      mode.lower(), color_mode.lower(), target_size)
    
    # Get labels
    labels = [label for label in os.Listdir(directory)
      if os.path.join(directory, label)]
    image_di.labels = labels.copy()

    # Create labels encoding and decoding
    for idx in range(len(labels)):
      # Decoding dict
      image_di.labels_decoding[idx] = labels[idx]
      # Encoding dict
      image_di.labels_encoding[labels[idx]] = idx
    
    # Files
    for root, dirs, files in os.walk(directory):
      # Get root folder as label
      current_folder_name = os.path.basename(root)
      # Converting data directory
      if current_folder_name in labels:
        # Get encoded label
        
        encoded_label = image_di.labels_encoding[current_folder_name]
        # Iterate all file in label folder
        for file in files:
          if encoded_label not in image_di.directory_data:
            image_di.directory_data[encoded_label] = []

          image_di.directory_data[encoded_label] \
            .append(os.path.join(root, file))
    
    # Shuffle data
    image_di.shuffle()
              
    return image_di