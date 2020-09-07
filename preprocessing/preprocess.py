# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import cv2

def preprocess_one_image(path_to_file):
  '''This function gets a single image ready to be fit into the neural network. 
  it converts the images from a input range of 0-255 to 0-1 for stability 
  during prediction.

  Image_requirments: Colored Image
  Image_output_shape: (124, 124, 3)
  
  
  
  parameters:
    - path_to_file: specifies the path to the input image -type:str

  rtype: numpy array

  Use case:
    img = preprocess_one_image('C:\\Users\\adewole opeyemi\\Downloads\\nudesdataset\\img1.jpg')
  '''


  try:
      img=Image.open(path_to_file)
      img=img.resize((124, 124))
      img = np.array(img, dtype='float16') / 255.
      if img.shape == (124, 124, 3):
        return img
      else:
        print("Image not multicolored, must be a multicolored image")
        return None
  except Exception as e:
    print(e)  
  return None

def preprocess_multiple_images(path_to_directory_of_images=None):
  '''This function gets a all images in a directory ready for training or testing into the neural network. 
  it converts the images from a input range of 0-255 to 0-1 for stability 
  during prediction.


  Images_output_shape: (number of colored images in directory ,124, 124, 3)
  
  
  parameters:
    - path_to_directory_of_images: specifies the path to the input image -type:str
    - path_to_save_npy: specifies the path to save images numpy array -type bool 

  rtype: numpy array


  Use case:
  img = preprocess_one_image('C:\\Users\\adewole opeyemi\\Downloads\\nudesdataset')
  
  '''
  
  listing = os.listdir(path=path_to_directory_of_images) 
  imgs = []
  for file in listing:
      try:
          img=Image.open(path_to_directory_of_images+'/'+file)
          img=img.resize((124, 124))
          img = np.array(img, dtype='float16') / 255.
          if img.shape == (124, 124, 3):
            imgs.append(img)

      except Exception as e:
        print(e)
        continue

  return np.array(imgs)

def preprocess_one_video(path_to_video):
  '''This function converts a video from a directory into frames which are then ready for training or testing into the neural network. 
  it converts the images from a input range of 0-255 to 0-1 for stability 
  during prediction.


  output_shape: (number of frames in the video ,124, 124, 3)
  
  
  parameters:
    - path_to_video: specifies the path to the input video -type:str
    

  rtype: numpy array


  Use case:
  img = preprocess_one_image('C:\\Users\\adewole opeyemi\\Downloads\\nudesdataset')
  
  '''

  cap = cv2.VideoCapture(path_to_video)
    
  frames=[]
  while True:
      ret, frame = cap.read()
      if ret == True:
          b = cv2.resize(frame,(124,124),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
          frames.append(b)
      else:
          break
      
  cap.release()
  cv2.destroyAllWindows()

  return np.array(frames)


def train_test_split(X, y, percent_split=0.9):
    assert len(X) == len(y), "input data must be the same size as targets"

    train_size = int(len(X) * percent_split)
    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:len(X)]
    y_test = y[train_size:len(y)]

    return ((X_train, y_train), (X_test, y_test))