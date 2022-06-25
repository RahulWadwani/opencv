#Write and execute programs for closing 
import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)
img= cv2.imread("car.jpg")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
closing = cv2.morphologyEx(imgGray, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closing img",closing)
cv2.waitKey(0)
