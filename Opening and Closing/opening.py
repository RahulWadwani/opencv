#Write and execute programs for opening 
import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)
img= cv2.imread("car.jpg")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
opening = cv2.morphologyEx(imgGray, cv2.MORPH_OPEN, kernel)
cv2.imshow("Opening img",opening)
cv2.waitKey(0)