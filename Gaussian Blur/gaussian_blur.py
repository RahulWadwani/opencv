#Write and execute programs for Dilation
import cv2
img= cv2.imread("Penguins.jpg")
imgGray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(img,(5,5),0)
cv2.imshow("img",img)
cv2.imshow("output",imgBlur)
cv2.waitKey(0)

