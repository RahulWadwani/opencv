from math import sqrt,exp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
img = cv2.imread("car 3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10,10))
plt.imshow(img,cmap=plt.cm.gray)
plt.axis('off')
#plt.show()
img = cv2.imread("car 3.jpg", 0)
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.subplot(151), plt.imshow(img,'gray'), plt.title('Original Image')

original = np.fft.fft2(img)
plt.subplot(152), plt.imshow(np.log(1+np.abs(original)), 'gray'), plt.title('Spectrum')

center = np.fft.fftshift(original)
plt.subplot(153), plt.imshow(np.log(1+np.abs(center)), 'gray'), plt.title('Centered Spectrum')

inv_center = np.fft.ifftshift(center)
plt.subplot(154), plt.imshow(np.log(1+np.abs(inv_center)), 'gray'), plt.title('Decentralized')

processed_img = np.fft.ifft2(inv_center)
plt.subplot(155), plt.imshow(np.abs(processed_img), 'gray'), plt.title('Processed Image')

plt.show()