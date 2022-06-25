#canny edge detection 
import cv2
import numpy as np
img = cv2.imread("Penguins.jpg",0)
canny_edge = cv2.Canny(img,100,100)
sigma =0.3
median = np.median(img)
lower= int(max(0,(1.0-sigma) * median))
upper= int(min(0,(1.0+sigma) * median))
auto_canny= cv2.Canny(img,lower,upper)
cv2.imshow("Canny",canny_edge)
cv2.imshow("Auto Canny",auto_canny)
cv2.waitKey(0)
