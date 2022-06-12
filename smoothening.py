
#Write and execute programs for smoothening filters  Spatial domain filtering 
import cv2
import numpy as np
img= cv2.imread("Penguins.jpg")
smoothed_mb = cv2.medianBlur(img,5)  
smoothed = cv2.GaussianBlur(img,(9,9),10)
cv2.imshow("smoothening effect",smoothed)
cv2.waitKey(0)