from loadData import loadData
import numpy as np

Directories = []
Directories.append("../DataSet01")
Directories.append("../Sample_Dataset")

images, measurements = loadData(Directories)
           
# for k in range(6):
#     k += 6*50
#     print(str(images[k].shape) + ', ' + str(measurements[k]))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Activation, Dropout, Convolution2D, MaxPooling2D

# Input, normalization & cropping
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,20), (0,0)))) # output size = (65, 320, 3)

# Convolutional layer #1
model.add(Convolution2D(filters = 14, kernel_size = 5, strides=1)) # output size = (62, 316, 14)
model.add(MaxPooling2D(pool_size=(2, 2))) # output size = (31, 158, 48)
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Convolutional layer #2
model.add(Convolution2D(filters = 36, kernel_size = 5, strides=1)) # output size = (13, 77, 36)
model.add(MaxPooling2D(pool_size=(2, 2))) # output size = (5, 35, 48)
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Convolutional layer #3
model.add(Convolution2D(filters = 48, kernel_size = 5, strides=1)) # output size = (10, 37, 48)
model.add(MaxPooling2D(pool_size=(2, 2))) # output size = (5, 35, 48)
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Convolutional layer #4
model.add(Convolution2D(filters = 64, kernel_size = 3, strides=1)) # output size = (3, 35, 64)
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Convolutional layer #5
model.add(Convolution2D(filters = 64, kernel_size = 3, strides=1)) # output size = (1, 33, 64)
model.add(Dropout(0.5))
model.add(Activation('relu'))

# Flatten and fully connected layers
model.add(Flatten())
model.add(Dense(100)) # fully connected layer
model.add(Dense(50)) # fully connected layer
model.add(Dense(10)) # fully connected layer
model.add(Dense(1)) # fully connected layer

# for layer in model.layers:
#     print(layer.output_shape)

# compile, fit and save
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=100)

model.save('model.h5')