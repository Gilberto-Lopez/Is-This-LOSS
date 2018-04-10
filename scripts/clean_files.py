# -*- coding: utf-8 -*-

import os

PNG = bytes([137,80,78,71,13,10,26,10])
JPG = bytes([0xFF,0xD8,0xFF])

def xx(x):
  y = x.replace('.jpg','.png')
  bx = open(x,'rb').read()
  if bx[:8] == PNG:
    with open(y,'wb') as z:
      z.write (bx)
    os.remove(x)

def ss(x):
  bx = open(x,'rb').read()
  if bx[:3] != JPG and bx[:8] != PNG:
    os.remove(x)

# Rename .jpg files that actually are .png
_,_,x = next(os.walk('./'))
list(map(xx,x))

# Remove non .jpg/.png files
_,_,x = next(os.walk('./'))
list(map(ss,x))
