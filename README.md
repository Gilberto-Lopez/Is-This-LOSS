# Is this LOSS?

Reconocimiento de Patrones y Aprendizaje Automatizado.

Proyecto 2: Sistema de Reconocimiento de objetos en imágenes.

## Objetivo

**Reconocimiento de Loss edits en imágenes.** Dado un conjunto de imágenes, principalmente memes, el objetivo es construir una red convolucional que aprenda a distinguir loss edits de aquellos memes que no lo son.

## LOSS

Para saber más de LOSS véase:

[LOSS en Know Your Meme](http://knowyourmeme.com/news/heres-to-loss-the-internets-greatest-meme "Here's to Loss, the Internet's Greatest Meme")

[Galería de LOSS en Know Your Meme](http://knowyourmeme.com/memes/loss/photos "Loss: Image Gallery")

[r/lossedits](https://www.reddit.com/r/lossedits/ "Loss Edits")

[Talking to the Man Behind 'Loss,' the Internet's Longest-Running Miscarriage 'Joke'](http://nymag.com/selectall/2015/11/longest-running-miscarriage-meme-on-the-web.html)

## Dataset

El dataset se obtuvo descargando edits de loss de la galería de Know Your Meme, buscando en Google Images (que contiene miniaturas de imágenes del subreddit y Know Your Meme, por lo que probablemente hay imágenes repetidas). Así como la colección personal de memes de los colaboradores.

Las imágenes fueron reescaladas para tener una resolución parecida a las miniaturas descargadas (al rededor de 250 * 250 pixeles).

### Estructura

La estructura del dataset es la siguiente:

```
dataset
  |
  +-> test_set
  |     |
  |     +-> loss_edits
  |     |
  |     +-> not_loss
  |
  +-> training_set
        |
        +-> loss_edits
        |
        +-> not_loss
```

donde `loss_edits` son las imágenes a distinguir de los memes vairados en `not_loss`.

### Descarga

El dataset se puede descargar del siguiente link: [dataset.7z](https://drive.google.com/file/d/1CztpddBSFyG22YZojRtke98S4OcHD9S7/view?usp=sharing "dataset.7z")

### Disclaimer

El dataset contiene los memes de los desarrolladores, una colección bastante edgy de dank memes variados, se recomienda discreción. Los memes contenidos no representan el punto de vista sobre la sociedad, la vida o temas sensibles por parte de los desarrolladores.

## Implementación

Las redes convolucionales se construyen en Python sobre [Keras](https://keras.io "Keras Documentation") con [TensorFlow](https://www.tensorflow.org/ "TensorFlow") como backend.

## Ejecución

Hay dos scripts: `CNN.py` para construir y entrenar una red convolucional y `TEST.py` para probar un modelo.

Sobre el directorio donde tenga el dataset y el script `cnn.py` ejecute el comando

### Construcción de la red

Con el script `CNN.py` se puede construir una red, se deben proporcionar los parámetros describiendo la arquitectura de la red. Puede obtener información
de los parámetros con el comando

```bash
$ python CNN.py -h
```

La red será entrenada con un dataset que siga la organización de directorios antes descrita.

Para construir el Modelo 1 descrito en el reporte del proyecto ejecute el siguiente comando:

```bash
$ python CNN.py -cl 2 -f 32 32 -d 5 3 -s 3 2 -dl 2 -a 64 64 -D dataset/ -o Modelo_1.h5
```

Para construir el Modelo 3:

```bash
$ python CNN.py -cl 2 -f 32 32 -d 5 3 -s 3 2 -dl 2 -a 64 64 -D dataset/ --RGB --optimizer adagrad -o Modelo_3.h5
```

### Probando la red

Con el script `TEST.py` se puede probar una red previamente entrenada con imágenes proporcionadas por el usuario. Puede obtener información
de los parámetros con el comando

```bash
$ python TEST.py -h
```

### Nota

Se recomienda ampliamente el uso de una tarjeta gráfica NVIDIA para realizar los cómpputos más rápido, esto si desea modificar los parámetros y/o agregar más capas al modelo.
