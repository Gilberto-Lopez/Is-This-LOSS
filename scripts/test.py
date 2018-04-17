# -*- coding: utf-8 -*-

# Load model
from keras.models import load_model

classifier = load_model('Keras_2C32_2D64_adagrad_rgb.h5')

# Test model with some images
import numpy as np
import cv2

color = cv2.COLOR_BGR2RGB

img = cv2.imread('loss_edit_1.jpg')
small = cv2.resize(img, (250, 250))
img_small = cv2.cvtColor(small, color)

img2 = cv2.imread('da_wae.jpg')
small2 = cv2.resize(img2, (250, 250))
img_small2 = cv2.cvtColor(small2, color)

img3 = cv2.imread('falls.jpeg')
small3 = cv2.resize(img3, (250, 250))
img_small3 = cv2.cvtColor(small3, color)

img4 = cv2.imread('zucc_erberg.jpeg')
small4 = cv2.resize(img4, (250, 250))
img_small4 = cv2.cvtColor(small4, color)

img5 = cv2.imread('loss_castle.jpg')
small5 = cv2.resize(img5, (250, 250))
img_small5 = cv2.cvtColor(small5, color)

img6 = cv2.imread('loss_monospace.png')
small6 = cv2.resize(img6, (250, 250))
img_small6 = cv2.cvtColor(small6, color)

img7 = cv2.imread('pose_loss.jpg')
small7 = cv2.resize(img7, (250, 250))
img_small7 = cv2.cvtColor(small7, color)

img8 = cv2.imread('loss_edit_2.jpeg')
small8 = cv2.resize(img8, (250, 250))
img_small8 = cv2.cvtColor(small8, color)

img9 = cv2.imread('loss_edit_3.jpeg')
small9 = cv2.resize(img9, (250, 250))
img_small9 = cv2.cvtColor(small9, color)

img10 = cv2.imread('moss.jpeg')
small10 = cv2.resize(img10, (250, 250))
img_small10 = cv2.cvtColor(small10, color)

img11 = cv2.imread('vector_loss.jpeg')
small11 = cv2.resize(img11, (250, 250))
img_small11 = cv2.cvtColor(small11, color)

img12 = cv2.imread('loss_tom.jpg')
small12 = cv2.resize(img12, (250, 250))
img_small12 = cv2.cvtColor(small12, color)

# import matplotlib.pyplot as plt

# plt.imshow(img_small)
# plt.show()

#cv2.imshow('loss',img_small)
#cv2.imwrite('img_small.png',img_small)

# RGB color

print ('\nLoss:',
  classifier.predict(np.expand_dims(img_small,axis=0)),
  '\nDa wae:',
  classifier.predict(np.expand_dims(img_small2,axis=0)),
  '\nFalls:',
  classifier.predict(np.expand_dims(img_small3,axis=0)),
  '\nZuckerberg:',
  classifier.predict(np.expand_dims(img_small4,axis=0)),
  '\nCastles:',
  classifier.predict(np.expand_dims(img_small5,axis=0)),
  '\nMonospace:',
  classifier.predict(np.expand_dims(img_small6,axis=0)),
  '\nPose:',
  classifier.predict(np.expand_dims(img_small7,axis=0)),
  '\nEdit2:',
  classifier.predict(np.expand_dims(img_small8,axis=0)),
  '\nEdit3:',
  classifier.predict(np.expand_dims(img_small9,axis=0)),
  '\nMoss:',
  classifier.predict(np.expand_dims(img_small10,axis=0)),
  '\nVectors:',
  classifier.predict(np.expand_dims(img_small11,axis=0)),
  '\nTom:',
  classifier.predict(np.expand_dims(img_small12,axis=0)))

# Grayscale

# print ('\nLoss:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small,axis=2),axis=0)),
#   '\nDa wae:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small2,axis=2),axis=0)),
#   '\nFalls:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small3,axis=2),axis=0)),
#   '\nZuckerberg:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small4,axis=2),axis=0)),
#   '\nCastles:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small5,axis=2),axis=0)),
#   '\nMonospace:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small6,axis=2),axis=0)),
#   '\nPose:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small7,axis=2),axis=0)),
#   '\nEdit2:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small8,axis=2),axis=0)),
#   '\nEdit3:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small9,axis=2),axis=0)),
#   '\nMoss:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small10,axis=2),axis=0)),
#   '\nVectors:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small11,axis=2),axis=0)),
#   '\nTom:',
#   classifier.predict(np.expand_dims(np.expand_dims(img_small12,axis=2),axis=0)))
