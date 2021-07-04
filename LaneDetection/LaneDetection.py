import numpy as np
import cv2 as cv


def ROI(frame,vertices) :
    mask=np.zeros_like(frame)
        
    match_mask_color=(255,)
    cv.fillPoly(mask,vertices,match_mask_color)
    masked=cv.bitwise_and(frame,mask)
    return masked

def drawlines(dilation) :
    lines = cv.HoughLinesP(dilation,2,np.pi/180,100,minLineLength=50,maxLineGap=10)
    for line in lines:
        if line is None :
            continue
        x1,y1,x2,y2 = line[0]
        cv.line(frame,(x1,y1),(x2,y2),(0,255,0),2,200)
    return frame


cap=cv.VideoCapture("/home/lajith/Documents/OpenCV/lane_vgt.mp4")

fourcc=cv.VideoWriter_fourcc(*'XVID')
out=cv.VideoWriter("Lane_detected.mp4",fourcc,20.0,(480,640))

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is TrueTrue
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    #Grayscale
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #cv.imshow("GRAY",gray)
 
    #Thresholding
    ret,thresh=cv.threshold(gray,175,255,cv.THRESH_TOZERO)
    #cv.imshow("Thresh",thresh)

    #Gaussian_Blurring
    G_blur=cv.GaussianBlur(thresh,(5,5),0)
    #cv.imshow("GBLUR",G_blur)

    #Canny Edge Detection
    canny=cv.Canny(G_blur,127,255)
    #cv.imshow("Canny",canny)




    height=frame.shape[0]
    width=frame.shape[1]

    #Defining Vertices for ROI
    ROI_vertices=[(0,height),(width,height),(width,height-300),(0,height-300)]

    


    ROI_image=ROI(canny,np.array([ROI_vertices],np.int32))

    kernel = np.ones((5,5),np.uint8)

    #Dilation
    dilation = cv.dilate(ROI_image,kernel,iterations = 1)
    #cv.imshow("DIL",dilation)

    #Finding lines and drawing them
    Lane_detected=drawlines(dilation)
    #Lane Detected Video
    cv.imshow("Lane Detected",Lane_detected)

    out.write(Lane_detected)


    if cv.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
cv.destroyAllWindows()




    
