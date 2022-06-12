#introductions 
#****************************************************************************************************
#opencv modules 
#core , imgproc , imgcodecs , highgui,video,calib3d
#features2d,objdetect,dnn,ml,flann,photo,stitching ,shape,superres,videostab,viz
#*************************************************************************************************
#accessing and manupulating pixels in opencv with BGR images 
#*************************************************************************************************
import cv2
import argparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

'''
import cv2
img = cv2.imread("car.jpg")
# Dimensions of an Image
dimensions= img.shape
# Size of the Image
total_number_of_elements=img.size
# Data type of Image
image_dtype=img.dtype
print(dimensions,total_number_of_elements,image_dtype)
(b,g,r)=img[6,40]
#pixel value accessed by row and column co-ordinates
print(b,g,r)
#Pixel value only one channel at a time 
b=img[0,40,0]
print(b)
#modification in the pixel values 
img[6,40]=(0,0,255)
print(img[6,40])
#top left corner of the image
top_left_corner=img[0:50,0:50]
print(top_left_corner)
cv2.imshow("corner",top_left_corner)
cv2.imshow("Original Image",img)
cv2.waitKey(0)
'''
#*************************************************************************************************
'''
import cv2
gray_img=cv2.imread("car.jpg",cv2.IMREAD_GRAYSCALE)
#gives the shape of the image 
dimensions =gray_img.shape
#get the value of the pixel(x=40,y=6)
i=gray_img[6,40]
#setting the pixel values to zero
gray_img[6,40]=0
print(i,gray_img[6,40],dimensions)
cv2.imshow("gray Image",gray_img)
cv2.waitKey(0)
'''
#***********************************************************************************************
#BGR order in Opencv
import cv2, numpy as np
import matplotlib.pyplot as plt
#Load image using cv2.imread
img_opencv=cv2.imread("car.jpg")
#split the loaded image into its three channels (b,g,r):
b,g,r=cv2.split(img_opencv)
#merger again the three channesl but in the RGB format
img_matplotlib = cv2.merge([r,g,b])
#show both images using matplotlib
#this will show the image in wrong colour 
plt.subplot(121)
plt.imshow(img_opencv)
#this will show the image in true color
plt.subplot(122)
plt.imshow(img_matplotlib)
plt.show()
#show both images using cv2.imshow(
# this will show the image in true color 
cv2.imshow('bgr image',img_opencv)
#this will show the image in wrong color
cv2.imshow('rgb image',img_matplotlib)
#to stack horizontally 
img_concats = np.concatenate((img_opencv,img_matplotlib),axis=1)
#now, we show the concatenated image:
cv2.imshow('bgr image and rgb image',img_concats)
#using numpy capabilitie sto get the channels and to builf the rgb image 
#get the three channels
B=img_opencv[:,:,0]
G=img_opencv[:,:,1]
R=img_opencv[:,:,2]
#Transform the image BGR to RGB  using Numpy capabilities 
img_matplotlib=img_opencv[:,:,::-1]
cv2.waitKey(0)
cv2.destroyAllWindows()
#**************************************************************************************************
#An introduction to handling files and images 
'''
import sys
print("The name of the script being processed is:'{}'".format(sys.argv[0]))
print("The name of the script being processed is:'{}'".format(len(sys.argv)))
print("The name of the script being processed is:'{}'".format(str(sys.argv)))
'''
#***************************************************************************************************
#argparse - command-line option and argument parsing 
'''
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("first_argument",help="this is the stirng text in connection with f")
args = parser.parse_args()
print(args.first_argument)
'''
#***************************************************************************************************
#Reading Images in OpenCv
'''
import argparse
import cv2
parser=argparse.ArgumentParser()
parser.add_argument("path_image",help="path to input image to be displayed")
args= parser.parse_args()
image=cv2.imread(args.path_image)
args= vars(parser.parse_args())
image2= cv2.imread(args["path_image"])
cv2.imshow("loaded image",image)
cv2.imshow("loaded image2",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#**************************************************************************************************
#converting to gray 
'''
import cv2
img = cv2.imread("car.jpg")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("output Gray image ",img_gray)
cv2.waitKey(0)
# another method to convert to gray 
import cv2
img2=cv2.imread("car.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("output image",img2)
cv2.waitKey(0)
'''
#***************************************************************************************************
#Reading camera frames and video files
#Reading images in Opencv
'''
import argparse
import cv2
parser= argparse.ArgumentParser()
parser.add_argument("path_image",help="path to input image to be displayed")
args= parser.parse_args()
image=cv2.imread(args.path_image)
args = vars(parser.parse_args())
image2 =cv2.imread(args["path_image"])
cv2.imshow("loaded image",image)
cv2.imshow("loaded image",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#Reading and writing images in Opencv
'''
import argparse
import cv2
parser = argparse.ArgumentParser()
parser.add_argument("path_image_input ",help="path to input image to be displayed ")
parser.add_argument("path_image_output ",help="path of the processed image to be saved ")
args= vars(parser.parse_args())
image_input=cv2.imread(args["path_image_input"])
cv2.imshow("loaded image",image_input)
gray_image=cv2.cvtColor(image_input,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray image",gray_image)
cv2.imwrite(args["path_image_output"],gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#Reading camera frames
'''
capture=cv2.VideoCapture(1)#1 is given to secondary camera input whereas 0 is given to primary camera or the webcam of your pc
'''
#accessing some properties of the capture object
'''
parser =argparse.ArgumentParser()
args =parser.parse_args()
capture =cv2.VideoCapture(1)
frame_width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps =capture.get(cv2.CAP_PROP_FPS)
print("CV_CAP_PROP_FRAME_WIDTH:'{}'".format(frame_width))
print("CV_CAP_PROP_FRAME_HEIGHT:'{}'".format(frame_height))
print("CAP_PROP_FPS:'{}'".format(fps))
if capture.isOpened() is False:
    print("Error in opening the camera")
while capture.isOpened():
    ret,frame =capture.read()
    if ret is True:
        cv2.imshow('input frame from the camera',frame)
        gray_frame = cv2.cvtcolor(frame,cv.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale  input camera',gray_frame)
        if cv2.waitKey(20) and 0xFF ==ord('q'):
            break
        else:
            break
capture.release()
cv2.destroyAllWindows()
'''
#Saving camera Frames
'''
if cv2.waitKey(20) & 0xFF == ord('c'):
    frame_name ="camera_frame_{}.png".format(frame_index)
    gray_frame_name ="grayscale_camera_frame_{}.png".format(frame_index)
    cv2.imwrite(frame_name,frame)
    cv2.imwrite(gray_frame_name,gray_frame)
    frame_index += 1
'''
#reading a video File
'''parser=argparse.ArgumentParser()
parser.add_argument("video_path",help="path to the video file")
args = parser.parse_args()
capture =cv2.VideoCapture(args.video_path)
'''
#Writing a video file
'''you can use cv2.VideoWriter'''
#calculation of frames per seconds 
'''
while capture.isOpened():
    ret, frame =capture.read()
    if ret is True:
        processing_start=time.time()
        processing_end=time.time()
        processing_time_frame=processing_end - processing_start
        print("fps:{}".format(1.0/processing_time_frame))
    else:
        break
processing_start=time.time()
processing_end=time.time()
processing_time_frame=processing_end - processing_start
print("fps:{}".format(1.0/processing_time_frame))
'''
#***********************************************************************************
#considerations for writing a video file
'''
parser = argparse.ArgumentParser()
parser.add_argument("output_video_path", help="path to the video file to write")
args = parser.parse_args()
capture = cv2.VideoCapture(0)
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_gray = cv2.VideoWriter(args.output_video_path, fourcc, int(fps), (int(frame_width),
while capture.isOpened():
    ret, frame =capture.read()
    if ret is True:
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        out_gray.write(gray_frame)
        cv2.imshow('gray',gray_frame)
        if cv2.waitKey(1)& 0xFF =ord('q'):
            break
    else:
        break
#playing with video capture properties
'''
'''you can use cv2.VideoCapture'''
#*************************************************************************************************
#getting all the propertied from the video capture object
'''
def decode_fourcc(fourcc):
    fourcc_int =int(fourcc)
    print("int value of fourcc :'{}'".format(fourcc_int))
    fourcc_decode=""
    for i in range (4):
        int_value =fourcc_int>>8 * i & 0xFF
        print("int_value:'{}'".format(int_value))
        fourcc_decode+=chr(int_value)
    return fourcc_decode
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS)))
print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))
print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))
print("CAP_PROP_FOURCC : '{}'".format(decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))))
print("CAP_PROP_FRAME_COUNT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
print("CAP_PROP_MODE : '{}'".format(capture.get(cv2.CAP_PROP_MODE)))
print("CAP_PROP_BRIGHTNESS : '{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS)))
print("CAP_PROP_CONTRAST : '{}'".format(capture.get(cv2.CAP_PROP_CONTRAST)))
print("CAP_PROP_SATURATION : '{}'".format(capture.get(cv2.CAP_PROP_SATURATION)))
print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE)))
print("CAP_PROP_GAIN : '{}'".format(capture.get(cv2.CAP_PROP_GAIN)))
print("CAP_PROP_EXPOSURE : '{}'".format(capture.get(cv2.CAP_PROP_EXPOSURE)))
print("CAP_PROP_CONVERT_RGB : '{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB)))
print("CAP_PROP_RECTIFICATION : '{}'".format(capture.get(cv2.CAP_PROP_RECTIFICATION)))
print("CAP_PROP_ISO_SPEED : '{}'".format(capture.get(cv2.CAP_PROP_ISO_SPEED)))
print("CAP_PROP_BUFFERSIZE : '{}'".format(capture.get(cv2.CAP_PROP_BUFFERSIZE)))
'''
#***************************************************************************************************************
#color triplets
'''
colors={'blue':(255,0,0),
'green':(0,255,0),
'red':(0,0,255),
'yellow':(0,255,255),
'magenta':(255,0,255),
'cyan':(255,255,0),
'dark_gray':(50,50,50)
}
colors['magenta']
image=np.zeros((500,500,3),dtype='uint8')
image[:]=colors['light_gray']
separation=40
for key in colors:
    cv2.line(image,(0,separation),(500,separation),colors[key],10)
    separation+=1

show_with_matplotlib(image,'dictionary with some prefered colors')
'''