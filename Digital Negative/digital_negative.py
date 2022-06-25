#digital negatives 
import cv2
img = cv2.imread("Penguins.jpg")
print(type(img))
Digital_negative = 255 - img
cv2.imshow("Digital negative image ",Digital_negative)
cv2.imshow("Original image",img)
cv2.waitKey(0)
