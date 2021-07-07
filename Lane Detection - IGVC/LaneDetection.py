import numpy as np
import cv2 
cap = cv2.VideoCapture('/home/lajith/Downloads/lane_vgt.mp4')

def ROI(frame,vertices) :
    mask=np.zeros_like(frame)
        
    match_mask_color=(255,)
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked=cv2.bitwise_and(frame,mask)
    return masked

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    height=frame.shape[0]
    width=frame.shape[1]
    #cv.imshow('frame', hsv)

    lower_hsv = np.array([0, 57, 126])
    higher_hsv = np.array([179, 96, 248])
    
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #cv2.imshow("Mask", mask)
    #cv2.imshow("Res",res)

    ret, thresh1 = cv2.threshold(mask, 197,255,cv2.THRESH_BINARY)
    cv2.imshow("Thresh", thresh1)

    
    kernel = np.ones((5,5),np.uint8)/25
    kernel2 = np.ones((7,7),np.uint8)/49
    kernel3 = np.ones((5,5),np.float32)/25
    kernel4 = np.ones((11,11),np.uint8)/121

    g_blur = cv2.GaussianBlur(thresh1,(5,5),0)
    cv2.imshow('g_blur',g_blur)

    
    
    closing = cv2.morphologyEx(g_blur, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closing',closing)

        
    dilation = cv2.dilate(closing,kernel,iterations = 1)
    cv2.imshow('dilation',dilation)
    
    erosion = cv2.erode(dilation, kernel2,iterations=1)

    canny = cv2.Canny(dilation,100,150,apertureSize=3)
    #cv2.imshow('canny',canny)


    ROI_vertices=[(0,height),(width,height),(width,height-300),(0,height-300)]

    ROI_image=ROI(canny,np.array([ROI_vertices],np.int32))

    lines=cv2.HoughLinesP(ROI_image,1,np.pi/180,40,20,20)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,0),5)
        cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()