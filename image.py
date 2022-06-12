#low-level processing image
import cv2
img=cv2.imread("Penguins.jpg",1)
#cv2.imshow("cute penguins",img)
print(img)
dimensions=img.shape
(b,g,r)=img[6,40]
print((b,g,r))
b,g,r = cv2.split(img)
print(b,g,r)
print(dimensions)
print(type(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
#mid-level processing
#basics of reading and displaying an image 
'''
import cv2
img =cv2.imread("Penguins.jpg",1)
img2 =cv2.imread("Penguins.jpg",0)#0 gives the grey scale image 
print(img)#gives the 3 dimensional array
print(type(img))#gives outputs the 
print(img.shape)
#displaying the image 
cv2.imshow("peguins",img2)
cv2.waitKey(0)#2000 represents the timer so that the image can be showed for 2000 millisec or second and automatically goes away
cv2.destroyAllWindows()#the moment we press any key it closes the image 
'''
#resizing of an image 
'''
import cv2
img =cv2.imread("Penguins.jpg",1)
resized = cv2.resize(img,(600,600))#this will resize the image to the particular values entered
#resized = cv2.resize(img,(int(img.shape[1]*2),int(img.shape[0]*2)))#this will double the size of the image 
#resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))#this will reduce the size of the image into half the size of the original image 
cv2.imshow("peguins ",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#write operation :- error could not find a writer for the specified extension
'''
import cv2
img = cv2.imread("Penguins.jpg",0)
resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow("penguins ",resized)
cv2.imwrite("penguins_resized",resized)
cv2.waitKey(200)
cv2.destroyAllWindows()
'''
import cv2
cap = cv2.VideoCapture(0)