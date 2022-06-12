from PIL import ImageGrab
import numpy as np
import datetime
import cv2
from win32api import GetSystemMetrics
width = GetSystemMetrics(0)
height = GetSystemMetrics(1)
time_stamp =datetime.datetime.now().strftime("%m-%d-%Y")
file_name=f'{time_stamp}.mp4'
fourcc =cv2.VideoWriter_fourcc('m','p','4','v')
captured_video =cv2.VideoWriter('output.mp4',fourcc, 20.0, (width,height))

while True:
    img =ImageGrab.grab(bbox=(0, 0, 1280, 720))
    img_np = np.array(img)
    img_final =cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    cv2.imshow('secret capture', img_final)
    captured_video.write(img_final)
    if cv2.waitKey(10) ==ord('q'):
        break
captured_video.release()
cv2.destroyAllWindows()