# -*- coding: utf-8 -*-

# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters = 32, kernel_size = (5, 5), strides = 3,
  input_shape = (250, 250, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = 2))

# Adding a second convolutional layer
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), strides = 2,
  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = 2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   #shear_range = 0.2,
                                   #zoom_range = 0.2,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  horizontal_flip = False)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (250, 250),
                                                 color_mode='grayscale',
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (250, 250),
                                            color_mode='grayscale',
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         #samples_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_set)
                         #nb_val_samples = 2000)

# Save trained model
classifier.save('Keras_2C32_2D64_adam_gray.h5')
