import cv2
import autopy 
import time
import HandTrackingModule as htm
import numpy as np

wcam , hcam = 1280,960
frameR = 100 #Frame Reduction
smoothening =10

plocX, plocY = 0,0
clocX, clocY = 0,0

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0
detector = htm.handDetector(maxHands=1)
wscr, hscr = autopy.screen.size()
#print(wscr, hscr)

while True:
    #1 Find Hand Landmarks
    
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    #print(lmList)
    #print(bbox)
    
    #2 Get the tip of index and middle fingers
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #print(x1, y1, x2, y2)
    

    #3 Check which fingers are up

        fingers = detector.fingersUp()
        #print(fingers)

        cv2.rectangle(img, (frameR, frameR),(wcam-frameR, hcam-frameR), (255,0,255),2)

    #4 Only Index finger : Moving mode

        if fingers[1]==1 and fingers[2]==0:

    #5 Convert coordinates (Convert into 640*480 coordinate system)
            
            x3 = np.interp(x1,[frameR,wcam-frameR],[0,wscr])
            y3 = np.interp(y1,[frameR,hcam-frameR],[0,hscr])
    #6 Smoothen Values
            clocX = plocX + (x3-plocX)/smoothening
            clocY = plocY + (y3-plocY)/smoothening



    #7 Move Mouse 
            autopy.mouse.move(wscr-clocX, clocY)
            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            plocX, plocY= clocX,clocY

    #8 Both index and middle fingers are up : Clicking Mode
        if fingers[1]==1 and fingers[2]==1:
    
    #9 Find distance between fingers
            
            length, img, lineInfo = detector.findDistance(8,12,img)
            #print(length)

    #10 Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4],lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                autopy.mouse.click()    
    #11 Frame Rate 
    cTime = time.time()
    fps = 1/(cTime - pTime)

    ptime = cTime
    cv2.putText(img, str(int(fps)), (20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    #12 Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1)==ord('s'):
        break
