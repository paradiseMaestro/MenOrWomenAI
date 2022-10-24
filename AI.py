import tensorflow as tf
import tensorflow.keras
import os
import keras
import numpy as np
# библиотека для вывода изображений
# import matplotlib.pyplot as plt
# %matplotlib inline
import cv2

from keras.models import Sequential

from keras.layers import Dense, Dropout , Conv3D , Flatten, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Activation
from keras.models import load_model

data = []
data1 = []
i = 2500 
while (i < 3000):
  
  data.append( 
      np.array(cv2.imread('./M/'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED))/255
  )
  data.append(
      np.array(cv2.imread('./W/'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED))/255
      )
  data1.append( [0., 1.])
  data1.append( [1., 0.])
  i = i + 1





# model = Sequential()
# model.add(tf.keras.layers.Conv2D( 2, 3, activation='relu', padding="same", input_shape=(299,299,1)))
# model.add(MaxPooling2D())
# model.add(tf.keras.layers.Conv2D( 2, 3, activation='relu', padding="same"))
# model.add(MaxPooling2D())
# # model.add(tf.keras.layers.Conv2D( 2, 3, activation='relu', padding="same"))
# # model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(512,activation='relu'))
# model.add(Dense(412,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(2,activation='softmax'))


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model = load_model('AI.h5')

model.fit(np.array(data), np.array(data1) ,      epochs=7 )

model.save('AI.h5')













# n = [

# ['500',' 1000'],
# ['1000',' 2000'],
# ['2000 ','3000'],
# ['3000','4000'],
# ['4000','5000'],
# ['5000','6000'],
# ['6000','7000'],
# ['7000','8000'],
# ['8000','9000'],
# ['9000','10000'],
# ['10000','11000'],
# ['11000','12000'],
# ['12000','13000'],
# ['13000','14000'],
# ['14000','15000'],
# ['15000','16000'],
# ['16000','17000'],
# ['17000','18000'],
# ['18000','19000'],
# ['19000','19550'],

# ]







# g=0
# while(g< len(n)):
#     model = load_model(modelName)
#     data = []
#     data1 = []
#     i = int(n[g][0])
#     while (int(n[g][0]) < int(n[g][1])):


#         data.append( 
#             np.array(cv2.imread('./v/'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED))/255
#         )
#         data.append(
#             np.array(cv2.imread('./Women/'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED))/255
#             )
#         data1.append( [0., 1.])
#         data1.append( [1., 0.])
#         i = i + 1

#     model.fit(np.array(data), np.array(data1) ,      epochs=5 )
#     model.save(modelName)
#     time.sleep(2)

#     g = g + 1

