# BUILDING THE CNN USING KERAS API

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras import optimizers
# from tensorflow.keras.utils import np_utils
# from keras import utils
import matplotlib
from shvn1 import train_data, train_label, test_data, test_label,test2,test2_l
import h5py

def create_model(width,height,channels):

     model = Sequential()
     model.add(keras.layers.Conv2D(16, (3, 3), input_shape=(width,height,channels), activation='relu',padding='same'))
     model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
     model.add(keras.layers.MaxPooling2D(pool_size=(6,6),strides= 6))
     model.add(keras.layers.Dropout(0.25))
     # model.add(keras.layers.Flatten())
     model.add(keras.layers.Conv2D(64,(5,9), activation='relu'))
     model.add(keras.layers.MaxPooling2D(pool_size = (2,2), strides =2))
     model.add(keras.layers.Conv2D(64,(2,4), activation = 'relu'))
     model.add(keras.layers.Dropout(0.25))
     model.add(keras.layers.Conv2D(128,(1,1),activation='relu'))
     model.add(keras.layers.Conv2D(1,(1,1), activation = 'sigmoid'))
     s=model.output_shape
     # model.add(keras.layers.MaxPooling2D(pool_size=(1,1)))
     model.add(keras.layers.Reshape((int(width/48),int(height/96)), input_shape = (int(width/48),int(height/96),1)))
     model.add(keras.layers.Flatten())
     model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])


     return model

test1 = create_model(48,96,1)
test1.summary()


print(train_data.shape)
test1.fit(train_data, train_label, validation_split = 0.1, epochs =5, verbose =2)
pred = test1.predict(test_data)
loss,accuracy=test1.evaluate(test_data, test_label,batch_size=32)
print('Test loss: %.4f accuracy: %.4f' % (loss, accuracy))


pred2=test1.predict(test2)
loss2, accuracy2= test1.evaluate(test2,test2_l)
print('Test loss: %.4f accuracy: %.4f' % (loss2, accuracy2))

first_2=np.argmax(pred, axis=1)[:1]
ans_2= (test_label)[:1]
print(first_2, "hi2")
print(ans_2, "hello2")

first_20=np.argmax(pred2)
ans_20= (test2_l)[:1]
print(first_20, "hi")
print(ans_20, "hello")


