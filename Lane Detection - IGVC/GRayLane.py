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


    height=frame.shape[0]
    width=frame.shape[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

    lower_hsv = np.array([0, 44, 126])
    higher_hsv = np.array([179, 96, 248])
    
    #ask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    #res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel5 = np.ones((3,3),np.uint8)/9
    kernel = np.ones((5,5),np.uint8)/25
    kernel2 = np.ones((7,7),np.uint8)/49
    kernel3 = np.ones((5,5),np.float32)/25
    kernel4 = np.ones((11,11),np.uint8)/121

    
    ret, thresh = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)

    cv2.imshow("thresh",thresh)

    g_blur = cv2.GaussianBlur(thresh,(7,7),0)
    cv2.imshow('g_blur',g_blur)

    
    
    closing = cv2.morphologyEx(g_blur, cv2.MORPH_CLOSE, kernel2)
    #cv2.imshow('closing',closing)

        
    dilation = cv2.dilate(closing,kernel,iterations = 1)
    #cv2.imshow('dilation',dilation)

    #erosion = cv2.erode(dilation, kernel, iterations=1)

    canny = cv2.Canny(dilation,100,150)
    cv2.imshow('canny',canny)

    ROI_vertices=[(0,height),(width,height),(width,height-300),(0,height-300)]

    ROI_image=ROI(canny,np.array([ROI_vertices],np.int32))

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(ROI_image, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    cv2.imshow("Lane",lines_edges)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()