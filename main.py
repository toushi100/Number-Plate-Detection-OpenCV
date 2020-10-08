import  cv2 as cv
import numpy as np

frameWidth = 640
frameHeight = 480
nPlateCascade = cv.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 200
color = (255,0,255)
count = 0

cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

while True:
    success, img = cap.read()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray,1.1,10)
    for x,y,w,h in numberPlates:
        area = w*h
        if area > minArea:
            cv.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv.putText(img," Number Plate",(x,y-5),cv.FONT_HERSHEY_COMPLEX,1,color,2)
            imgRoi = img[y:y+h,x:x+w]
            cv.imshow("ROI",imgRoi)
    cv.imshow("result",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
       cv.imwrite("Number Plate"+str(count)+".jpg",imgRoi)
       cv.rectangle(img,(0,200),(640,300),color,cv.FILLED)
       cv.putText(img,"Image Saved",(150,265),cv.FONT_HERSHEY_COMPLEX,2,color,2)
       cv.imshow("Result",img)
       cv.waitKey(5000)
       count+=1
