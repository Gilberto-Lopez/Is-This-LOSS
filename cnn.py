# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

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
classifier.add(Conv2D(filters = 64, kernel_size = (5, 5), strides = 2,
  input_shape = (250, 250, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = 3))

# Adding a second convolutional layer
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), strides = 1,
  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = 2))

classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), strides = 1,
  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = 2))




# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(128, activation = 'relu'))
#classifier.add(Dense(128, activation = 'relu'))

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
                                                 color_mode='rgb',
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (250, 250),
                                            color_mode='rgb',
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         #samples_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_set)
                         #nb_val_samples = 2000)
