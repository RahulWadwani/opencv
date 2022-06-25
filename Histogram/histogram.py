#write and execute programs for image equalization of histogram and histogram function 
import cv2
import numpy as np
import matplotlib.pyplot as plt
img= cv2.imread("Penguins.jpg",0)
hist,bins= np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalised = cdf * float(hist.max())/cdf.max()
plt.plot(cdf_normalised,color = 'b')
plt.hist(img.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'),loc='upper left')
plt.show()
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))
cv2.imshow('equ.jpg',equ)
cv2.waitKey(0)
