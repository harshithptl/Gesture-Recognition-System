import cv2 
import random
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Convolution2D,Dropout,Activation,Flatten,MaxPooling2D,ZeroPadding2D,Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


#Specifying project path.Change to your needs where your images are stored.Add all gesture categories
data_dir='C:\Personal\Projects\Gesture Recognition'
categories=['Fist','Five','Shaka','One','Two','None']

train=[]


for category in categories:
    path=os.path.join(data_dir,category) 
    
    #Assigning index value as label for each gesture.
    class_num=categories.index(category) 
    
    #Creating training data by reading each image and label.
    for img in os.listdir(path): 
        img_array=cv2.imread(os.path.join(path,img))            
        train.append([img_array,class_num])
    

#Shuffling data so that training and test data is not imbalanced.     
random.shuffle(train)

#Creating training dataset and target labels
X=[]
y=[]

for features,label in train:
    X.append(features)
    y.append(label)

X=np.array(X)

#Normalising training data and one hot encoding y so that all classes are represented by 1's and 0's
X=np.array(X/255.0)
y=to_categorical(y)
y=np.array(y)
print(X.shape,y.shape)

#Building Model.Uses VGG net architecture.

custom_vgg = Sequential()
custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = (50, 50, 3)))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Flatten())
custom_vgg.add(Dense(6, activation = "softmax"))

custom_vgg.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

custom_vgg.fit(X,y,epochs=7,validation_split=0.1)
    
custom_vgg.save('model.h5')