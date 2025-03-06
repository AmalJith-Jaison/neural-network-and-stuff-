import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import MaxPooling2D,Conv2D,Flatten,Dense
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)
xtrain=xtrain.reshape(60000,28,28,1).astype('float')/255.0
xtest=xtest.reshape(10000,28,28,1).astype('float')/255.0
model2=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
    
])
model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model2.fit(xtrain,ytrain,epochs=15,validation_split=0.2,batch_size=64)
import matplotlib.pyplot as plt
plt.plot(m1history.history['accuracy'],label='accuracy')
plt.plot(m1history.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()
