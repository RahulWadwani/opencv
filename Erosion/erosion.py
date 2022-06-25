#Write and execute programs for Erosion 
import cv2
import numpy as np
img= cv2.imread("Penguins.jpg")
kernel = np.ones((5,5),np.uint8)
imgCanny = cv2.Canny(img,100,100) 
imgDilated=cv2.dilate(imgCanny,kernel,iterations=1)
imgErode= cv2.erode(imgDilated,kernel,iterations=1)
cv2.imshow("Eroded Image", imgErode)
cv2.waitKey(0)

