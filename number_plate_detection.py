#IMAGE PROCESSING
#Basic Operations on Image and Neighbourhood Processing
# converting Coloured Image to Gray Scale Image
'''import cv2                                      #Importing the opencv library
img=cv2.imread("car.jpg")                       #Reading an Image
cv2.imshow("Original Image",img)                #Showing the original Image 
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #Converting Original Image to Gray Image
cv2.imshow("Gray Image",imgGray)                #Showing the Image
cv2.waitKey(0)                                  #Adding a Delay 
#Edge Detection of an Image
import cv2                          #Importing opencv Library
img=cv2.imread("car.jpg")           #Reading an Image
imgCanny=cv2.Canny(img,100,100)     #Detecting Egdes using Canny Operations
cv2.imshow("Canny Image",imgCanny)  #Showing the Edges of the Image 
cv2.waitKey(0)                      #Adding a Delay
#Converting an Image to Blurred Image
import cv2                                        #Importing opencv Library
img=cv2.imread("car.jpg")                         #Reading an Image
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)      #Converting RGB Image to Gray 
imgBlurred=cv2.GaussianBlur(imgGray,(7,7),0)      #Detecting Egdes using Canny Operations
cv2.imshow("Blurred Image",imgBlurred)              #Showing the Edges of the Image 
cv2.waitKey(0)                                    #Adding a Delay
'''
#Performing Erosion on an Image
'''import cv2                                          #Importing opencv Library
import numpy as np                                  #Importing Numpy Library
img=cv2.imread("car.jpg")                           #Reading an Image
kernel=np.ones((5,5),np.uint8)                      #Creating a mask of 5x5 matrix  
imgCanny=cv2.Canny(img,100,100)                     #Converting RGB Image to Gray 
imgDilated=cv2.dilate(imgCanny,kernel,iterations=1) #Dilating an Image
imgErode=cv2.erode(imgDilated,kernel,iterations=1)  #Erosion Image
cv2.imshow("Eroded Image",imgErode)                 #Showing the Edges of the Image 
cv2.waitKey(0)                                      #Adding Delay to Image
'''
#Performing Dilation on an Image
'''import cv2                                          #Importing opencv Library
import numpy as np                                  #Importing Numpy Library
img=cv2.imread("car.jpg")                           #Reading an Image
kernel=np.ones((5,5),np.uint8)                      #Creating a mask of 5x5 matrix  
imgCanny=cv2.Canny(img,100,100)                     #Converting RGB Image to Gray 
imgDilated=cv2.dilate(imgCanny,kernel,iterations=1) #Dilating an Image
cv2.imshow("Dilated Image",imgDilated)              #Showing the Edges of the Image 
cv2.waitKey(0)                                      #Adding Delay to Image
'''
#Digital Negative Image
'''import cv2                          # importing the opencv library
img= cv2.imread("car.jpg")          # reading the image
imgNeg = 255 - img                  # using the formula (l-1)-r to show the digital negative of an image
cv2.imshow("Negative Image",imgNeg) # showing the Negative Image
cv2.waitKey(0)                      # Delay 
'''
#Resizing an Image
'''import cv2                                          #Importing opencv Library
import numpy as np                                  #Importing Numpy Library
img=cv2.imread("car.jpg")                           #Reading an Image
print(img.shape) 
imgResize=cv2.resize(img,(300,200))  #Erosion Image
cv2.imshow("Resized Image",imgResize)                 #Showing the Edges of the Image 
cv2.waitKey(0)                                      #Adding Delay to Image
'''
#Cropping an Image
'''import cv2                                          #Importing opencv Library
import numpy as np                                  #Importing Numpy Library
img=cv2.imread("car.jpg")                           #Reading an Image
imgCropped=img[:200,200:400]
cv2.imshow("Cropped Image",imgCropped)                 #Showing the Edges of the Image 
cv2.waitKey(0)                                      #Adding Delay to Image
#Rotating an image
import cv2                                          #Importing opencv Library
import numpy as np                                  #Importing Numpy Library
img=cv2.imread("car.jpg")                           #Reading an Image
print(img.shape) 
imgRotate=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)  #Erosion Image
cv2.imshow("Resized Image",imgResize)                 #Showing the Edges of the Image 
cv2.waitKey(0)                                      #Adding Delay to Image
'''
# map 
'''n = int(input())
arr = list((int, input().split()))
z = arr.sort(reverse=True)
print(z[1])
'''
'''records=[]
for _ in range(int(input())):
    name= str(input())
    score= float(input())
    records.append([name,score])
sorted_scores=sorted(list(set(x[1] for x in records)))
second_lowest=sorted_scores[1]

low_Final_list=[]
for student in records:
    if second_lowest == student[1]:
        low_Final_list.append(student[0])
for student in sorted(low_Final_list):
    print(student)
print("hello")
'''

#*****************************************************************************************
#number plate detection 
'''
import numpy as np
import cv2
from PIL import Image 
import pytesseract as tess
def ratiocheck(area,width,height):
    ratio = float(width)/float (height)
    if ratio <1:
        ratio =1/ratio
    if (area <1063.62 or area>73862.5) or (ratio<3 or ratio >6):
        return False
    return True
def isMaxWhite(plate):
    avg =  np.mean(plate)
    if avg>=115:
        return True
    else:
        return False
def ratio_and_rotation (rect):
    (x,y),(width,height),rect_angle=rect
    if width>height:
        angle=-rect_angle
    else:
        angle = 90 + rect_angle
    if angle>15:
        return False 
    if height == 0 or width == 0:
        return False 
    area = height * width 
    if not ratiocheck(area,width,height):
        return False
    else:
        return True

def clean2_plate(plate):
    gray_img=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.cvtColor(gray_img, 110, 255, cv2.THRESH_BINARY)
    if cv2.waitKey(0) and 0xff == ord ('q'):
        pass
    num_contours, heirarchy =cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    if num_contours :
        contour_area=[cv2.contourArea(c)for c in num_contours]
        max_cntr_index= np.argmax(contour_area)

        max_cnt =num_contours[max_cntr_index]
        max_cntArea = contour_area[max_cntr_index]
        x,y,w,h= cv2.boundingRect(max_cnt)

        if not ratiocheck(max_cntArea,w,h):
            return plate,None

        final_img = thresh[y:y+h,x:x+w]
        return final_img,[x,y,w,h]
    else:
        return plate,None
img =cv2.imread("car 4.jpg")
#print("Number input image ")
cv2.imshow("input",img)
if cv2.waitKey(0) & 0xff==ord('q'):
   pass
img2 = cv2.GaussianBlur(img,(3,3),0)
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

img2= cv2.Sobel(img2,cv2.CV_8U,1,0,ksize = 3)
_,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

element = cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(17,3))
morph_img_threshold=img2.copy()
cv2.morphologyEx(src=img2,op=cv2.MORPH_CLOSE,kernel=element,dst= morph_img_threshold)
num_contours,hierarchy= cv2.findContours(morph_img_threshold,mode= cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img2,num_contours, -1, (0,255,0), 1)



for i,cnt in enumerate(num_contours):
    min_rect=cv2.minAreaRect(cnt)

    if ratio_and_rotation(min_rect):
        x,y,w,h = cv2.boundingRect(cnt)
        plate_img =img[y:y+h,x:x+w]
        print("Number identified number plate ...")
        cv2.imshow("num plate image",plate_img)
        if cv2.waitKey(0)& 0xff== ord('q'):
            pass
        if (isMaxWhite(plate_img)):
            clean_plate,rect=clean2_plate(plate_img)
            if rect:
                fg=0
                x1,y1,w1,h1=rect
                x,y,w,h=x+x1,y+y1,w1,h1
                plate_im = Image.fromarray(clean_plate)
                text=tess.image_to_string(plate_im,lang='eng')
                print("number detected plate text:",text)
'''
#number plate detection 
#importing the libraries 
import cv2
import imutils
#from PIL import Image
import pytesseract 
#file path redirection 
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Rahul\AppData\Local\Tesseract-OCR\tesseract-ocr-w64-setup-v5.0.1.20220118.exe"
#importing the image and by using the imutils functions we resize the image 
image=cv2.imread("car 3.jpg")

image = imutils.resize(image,width=300)
cv2.imshow("original image",image)
cv2.waitKey(0)
#converting to grayscale image 
imageGray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray image",imageGray)
cv2.waitKey(0)
#reducing the noise in the image 
gray_image = cv2.bilateralFilter(imageGray,11,17,17)#filter size 
cv2.imshow("smoothened image",gray_image)
cv2.waitKey(0)
#detecting the edges using Canny edge detection 
edged= cv2.Canny(gray_image,30,200)#threshold values 
cv2.imshow("edges image",edged)
cv2.waitKey(0)
#finding the contours in an image 
cnts,new= cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)#retr retrieves all the contours but doesnt create any parent child 
#chain approx removes the redundant points
image1=image.copy()
cv2.drawContours(image1,cnts,-1,(0,255,0),3)
cv2.imshow("contours",image1)
cv2.waitKey(0)
#sorting identified 
cnt = sorted(cnts,key=cv2.contourArea,reverse=True)[:30]
screenCnt = None
image2 = image.copy()
cv2.drawContours(image2,cnts,-1,(0,255,0),3)
cv2.imshow("top 30 images ",image2)
cv2.waitKey(0)
#finding the contour with four side 
i= 7 
for c in cnts:
    perimeter = cv2.arcLength(c,True) # 
    approx = cv2.approxPolyDP(c,0.018*perimeter,True) #approximates the curve polygon 
    if len(approx) == 4:
        screenCnt = approx
        x,y,w,h = cv2.boundingRect(c)
        new_img = image [y:y+h,x:x+w]
        cv2.imwrite('./'+str(i)+'.png',new_img)#
        i+=1
        break
cv2.drawContours(image,[screenCnt],-1,(0,255,0),3)
cv2.imshow("image with detected license plate ",image)
cv2.waitKey(0)
cropped_loc = './7.jpg'
cv2.imshow("cropped ",cv2.imread(cropped_loc))
plate = pytesseract.image_to_string(cropped_loc,lang = 'eng')
print("number plate is:",plate)
cv2.waitkey(0)
cv2.destroyAllWindows()
