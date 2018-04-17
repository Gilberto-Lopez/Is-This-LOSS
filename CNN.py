#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-cl','--clayers', type=int, default=2, metavar='N',
  help='Número de capas de convolución. Por defecto 2')
parser.add_argument('-f','--filters', type=int, required=True,
  metavar='N', nargs='+',
  help='Número de filtros')
parser.add_argument('-d','--dimension', type=int,
  metavar='D', nargs='+', required=True,
  help='Dimensión de los filtros D*D')
parser.add_argument('-s','--strides', type=int,
  metavar='N', nargs='+', required=True,
  help='Strides o pasos de los filtros')
parser.add_argument('-dl','--dlayers', type=int, default=2, metavar='N',
  help='Número de capas del clasificador. Por defecto 2')
parser.add_argument('-a','--arch', type=int, required=True,
  metavar='NODES', nargs='+',
  help='Arquitectura del clasificador, nodos en la(s) capa(s) oculta(s)')
parser.add_argument('--optimizer', default='adam', choices=['adam,adagrad'],
  help='El optimizador para el entrenamiento del modelo. Por defecto "adam"')
parser.add_argument('-D','--dataset', required=True, metavar='DIR',
  help='El directorio con el dataset para entrenar el modelo')
parser.add_argument('--RGB', action='store_true',
  help='Si el modelo trabajará con imágenes a color RGB o escala de grises')
parser.add_argument('-e','--epochs', type=int, default=10,
  help='El número de épocas a entrenar el modelo. Por defecto 10')
parser.add_argument('-o','--outfile', default='Model.h5',
  help='El nombre de archivo para guardar el modelo. Por defecto "Modelo.h5"')

args = parser.parse_args()

if args.clayers < 1:
  raise ValueError('El número de capas de convolución debe ser un número positivo: {}'
    .format(args.clayers))
if len(args.filters) != args.clayers:
  raise ValueError('La cantidad de filtros dados no coincide con la cantidad de capas: {} != {}'
    .format(args.clayers,len(args.filters)))
for f in args.filters:
  if f < 1:
    raise ValueError('La cantidad de filtros en cada capa debe ser un número positivo: {}'
      .format(f))
if len(args.dimension) != args.clayers:
  raise ValueError('La cantidad de dimensiones de filtros dadas no coincide con la cantidad de capas: {} != {}'
    .format(args.clayers,len(args.dimension)))
for dim in args.dimension:
  if dim < 1:
    raise ValueError('Las dimensiones de los filtros deben ser números positivos: {}'
      .format(dim))
if len(args.strides) != args.clayers:
  raise ValueError('La cantidad de strides dados no coincide con la cantidad de capas: {} != {}'
    .format(args.clayers,len(args.strides)))
for stride in args.strides:
  if stride < 1:
    raise ValueError('Los strides los filtros deben ser números positivos: {}'
      .format(stride))
if args.dlayers < 1:
  raise ValueError('El número de capas ocultas del clasificador debe ser un número positivo: {}'
    .format(args.dlayers))
if len(args.arch) != args.dlayers:
  raise ValueError('La cantidad de nodos por en capas ocultas no coincide con la cantidad de capas del clasificador: {} != {}'
    .format(args.clayers,len(args.arch)))
for nodes in args.arch:
  if nodes < 1:
    raise ValueError('El número de nodos en las capas ocultas debe ser un número positivo: {}'
      .format(nodes))
if args.epochs < 1:
  raise ValueError('El número de épocas debe ser un número positivo: {}'
    .format(args.epochs))
if args.dataset[-1] != os.sep:
  args.dataset += os.sep

#############################
#  Construcción del modelo  #
#############################

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

# Primera capa de convolución
classifier.add(Conv2D(filters = args.filters[0],
  kernel_size = args.dimension[0],
  strides = args.strides[0],
  input_shape = (250, 250, 3 if args.RGB else 1),
  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = 2))

# Capas de convolución restantes
for i in range(1,args.clayers):
  classifier.add(Conv2D(filters = args.filters[i],
    kernel_size = args.dimension[i],
    strides = args.strides[i],
    activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = 2))

# "Aplanar" los mapas de características reducidos
classifier.add(Flatten())

# Clasificador
for i in range(args.dlayers):
  classifier.add(Dense(args.arch[i], activation = 'relu'))

# Salida del clasificador
classifier.add(Dense(1, activation = 'sigmoid'))

# Compilar el modelo
classifier.compile(optimizer = args.optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

##############################
#  Datos para entrenamiento  #
##############################

train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = False)

training_set = train_datagen.flow_from_directory(args.dataset+'training_set',
  target_size = (250, 250),
  color_mode='rgb' if args.RGB else 'grayscale',
  batch_size = 32,
  class_mode = 'binary')

test_set = test_datagen.flow_from_directory(args.dataset+'test_set',
  target_size = (250, 250),
  color_mode='rgb' if args.RGB else 'grayscale',
  batch_size = 32,
  class_mode = 'binary')

##############################
#  Entrenamiento del modelo  #
##############################

classifier.fit_generator(training_set,
  epochs = args.epochs,
  validation_data = test_set)

#######################
#  Guardar el modelo  #
#######################

classifier.save(args.outfile)
