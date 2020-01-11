# Lane-Detection-Opencv
This repository consists of program for Lane detection using Opencv.

#.....................Functions..............................#
import cv2
import numpy as np
import matplotlib.pyplot as plt


def pyploter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                      # converting color 2 gray img
    blur = cv2.GaussianBlur(gray,(5,5),0)                            # making it blur to avoid noise..
    canny_edge = cv2.Canny(blur,50,150)                              # finding edges with high change in intensity
    return canny_edge

# finding edges inside roi
def region_O_interest(img):
    height= img.shape[0]
    width = img.shape[1]
    polygons = np.array([[(0,height),(width,height),(370,30)]])       # marking co-ordinates for roi
    mask = np.zeros_like(img)                                        # masking entire image with black
    cv2.fillPoly(mask,polygons,(255,255,255))                        # roi white
    roi = cv2.bitwise_and(img,mask)                                  # Bitwise-and operation
    return roi


# displaying generated lines in green color
def Display_image(img, lines):
    line_image = np.zeros_like(img)                                    # masking image with black color
    if lines is not None:
        for x1,y1,x2,y2 in lines:                                      # for every line in lines
            # x1,y1,x2,y2 = line.reshape(4)                            # getting co-ordinates of the line
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)          # marking the lines with green color over the masked image
    return line_image


def co_ord(img,parameters):
    slope,intercept = parameters
    y1 = img.shape[0]
    y2 = int(y1*(2/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1,y1,x2,y2])

def average_slope(img,lines):
    left_lane=[]
    right_lane=[]
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameter = np.polyfit((x1,x2),(y1,y2),1)
            slope = parameter[0]
            intercept = parameter[1]
            if slope > 0:
                left_lane.append((slope, intercept))
            else:
                right_lane.append((slope, intercept))
    left_lane_avg = np.average(left_lane,axis = 0)
    right_lane_avg = np.average(right_lane, axis=0)
    left_line = co_ord(img,left_lane_avg)
    right_line = co_ord(img,right_lane_avg)


    return np.array([left_line,right_line])



#########################################PROGRAM####################################################
original_img = cv2.imread('./lane.jpeg')
lane_image = np.copy(original_img)
roi_image = region_O_interest(pyploter(lane_image))
lines = cv2.HoughLinesP(roi_image,2,np.pi/180,100,np.array([]),20,5) # finding lines in roi
avg_line = average_slope(lane_image,lines)
line_image = Display_image(lane_image,avg_line)

combo_img = cv2.addWeighted(lane_image,0.7,line_image,0.9,1) # combining line-img with lane-img
cv2.imshow('LANE DETECTED', combo_img)
cv2.waitKey(0)
