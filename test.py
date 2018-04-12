# -*- coding: utf-8 -*-

# Load model
from keras.models import load_model

model = load_model('Keras_2C32_2D64_adam_gray.h5')

# Test model with some images
import numpy as np
import cv2

img = cv2.imread('loss_edit_1.jpg')
small = cv2.resize(img, (250, 250))
gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('da_wae.jpg')
small2 = cv2.resize(img2, (250, 250))
gray_small2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

img3 = cv2.imread('falls.jpeg')
small3 = cv2.resize(img3, (250, 250))
gray_small3 = cv2.cvtColor(small3, cv2.COLOR_BGR2GRAY)

img4 = cv2.imread('zucc_erberg.jpeg')
small4 = cv2.resize(img4, (250, 250))
gray_small4 = cv2.cvtColor(small4, cv2.COLOR_BGR2GRAY)

# import matplotlib.pyplot as plt

# plt.imshow(gray_small)
# plt.show()

#cv2.imshow('loss',gray_small)
#cv2.imwrite('gray_small.png',gray_small)

print ('Loss:',
  classifier.predict(np.expand_dims(np.expand_dims(gray_small, axis = 2),axis=0)),
  'Da wae:',
  classifier.predict(np.expand_dims(np.expand_dims(gray_small2, axis = 2),axis=0)),
  'Falls:',
  classifier.predict(np.expand_dims(np.expand_dims(gray_small3, axis = 2),axis=0)),
  'Zuckerberg:',
  classifier.predict(np.expand_dims(np.expand_dims(gray_small4, axis = 2),axis=0)))
