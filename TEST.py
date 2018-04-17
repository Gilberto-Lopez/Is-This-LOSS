#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-M','--model', default='Modelo.h5',
  metavar='FILE', required=True,
  help='Modelo entrenado')
parser.add_argument('-i','--image', required=True,
  metavar='IMAGE', nargs='+',
  help='Imágenes para probar el modelo entrenado')
parser.add_argument('--RGB', action='store_true',
  help='Si el modelo fue entrenado con imágenes a color RGB o escala de grises')

args = parser.parse_args()

######################
#  Cargar el modelo  #
######################

from keras.models import load_model

classifier = load_model(args.model)

######################
#  Probar el modelo  #
######################

import numpy as np
import cv2

color = cv2.COLOR_BGR2RGB if args.RGB else cv2.COLOR_BGR2GRAY

for img_name in args.image:
  img = cv2.imread(img_name)
  small = cv2.resize(img, (250, 250))
  img_small = cv2.cvtColor(small, color)
  if not args.RGB:
    img_small = np.expand_dims(img_small,axis=2)
  img_small = np.expand_dims(img_small,axis=0)

  print ('\nImagen: "{}"\tClasificación: "{}"'
    .format(img_name,
      'loss_edit' if classifier.predict(img_small) == 0 else 'not_loss'))


