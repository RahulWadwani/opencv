'''import cv2 as cv
cap = cv.VideoCapture(0)
address="https://192.168.0.102:8080/video"
cap.open(address)
while cap.isOpened():
    ret,frame = cap.read()
    cv.imshow('Holistic Model Detections',frame)
    if cv.waitKey(10) and 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()'''
print("hello")