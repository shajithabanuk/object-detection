import cv2
import imutils
cam = cv2.VideoCapture(0)
fristFrame = None
area = 500
while True:
    _, img = cam.read()
    text = "Normal"
##    if not _ or img is None:
##        print("failed to grab frame")
##        continue
    img = imutils.resize(img, width=1000) 
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg,(21,21),0)
    if fristFrame is None:
        fristFrame = gaussianImg
        continue
    imgDiff = cv2.absdiff(fristFrame,gaussianImg)
    threshImg = cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg,None,iterations=2)
    cnts = cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        text = "Moving object Detected"
    print(text)
    cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imshow("cameraFeed",img)
    key=cv2.waitKey(1)
    if  key==ord("a"):
        break
cam.release()
cv2.destroyAllWindows()
