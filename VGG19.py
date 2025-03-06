import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten,Dense
from keras.applications import VGG19
from keras.utils import to_categorical
import numpy as np
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
xtest=np.repeat(tf.image.resize(xtest[...,np.newaxis],(32,32)).numpy(),3,axis=-1)
xtrain=np.repeat(tf.image.resize(xtrain[...,np.newaxis],(32,32)).numpy(),3,axis=-1)
xtrain=xtrain.astype('float')/255.0
xtest=xtest.astype('float')/255.0
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)
base=VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(32,32,3)
    )
base.trainable=False
model=Sequential()
model.add(base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model.fit(xtrain,ytrain,epochs=15,validation_split=0.2,batch_size=64)
