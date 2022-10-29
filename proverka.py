import os
import cv2
import numpy as np

from keras.models import Sequential
import tensorflow as tf
import tensorflow.keras
import os
import keras
from keras.layers import Dense, Dropout , Conv3D , Flatten, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Activation
from keras.models import load_model


def more(h):
  numbers = map(np.float32, h)
  index, max_value = max(enumerate(numbers), key=lambda i_v: i_v[1])
  return index

model = load_model('AI.h5')
# i = 0
# n = model.predict(
#     np.array(
# [np.array(cv2.imread('./W/'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED))/255]
#     )
# )
# print(more(n))



# def more(h):
#   numbers = map(np.float32, h)
#   index, max_value = max(enumerate(numbers), key=lambda i_v: i_v[1])
#   return index

i = 58884
while ( i < 81982):
    n = model.predict(
        np.array(
    [np.array(cv2.imread('./h/'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED))/255]
        )
    )

    if (more(n[0])==1):
        os.rename('./'+'NewPic'+'/'+str(i)+'.jpg','./'+'M'+'/'+str(i)+'.jpg')
    i=i+1






