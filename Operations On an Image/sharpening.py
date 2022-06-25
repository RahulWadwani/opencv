#Write and execute programs for sharpening filter spatial domain filtering 
import cv2
import numpy as np 
img= cv2.imread("Penguins.jpg")
smoothed_mb=cv2.medianBlur(img,5)
smoothed = cv2.GaussianBlur(img,(9,9),10)
sharped = cv2.addWeighted(smoothed_mb,1.5,smoothed,0,-0.5,0)
cv2.imshow("sharpening Filters",sharped)
cv2.waitKey(0)

