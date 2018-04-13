import cv2, numpy as np, matplotlib.pyplot as plt

img = cv2.imread('falls.jpeg')
img = cv2.resize(img, (250, 250))
edges1 = cv2.Canny(img,100,200)
edges2 = cv2.Canny(img,350,500)

plt.subplot(131),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges1,cmap = 'gray')
plt.title('Aristas con ruido'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(edges2,cmap = 'gray')
plt.title('Aristas con pérdida'), plt.xticks([]), plt.yticks([])
plt.show()
