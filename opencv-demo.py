# -*- coding:utf-8 -*-
__author__ = 'lincolnfz@gmail.com'
import cv2  
import numpy as np

def canny_1():
  
    def CannyThreshold(lowThreshold):  
        detected_edges = cv2.GaussianBlur(gray,(3,3),0)  
        detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)  
        dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.  
        cv2.imshow('canny demo',dst)  
      
    lowThreshold = 0  
    max_lowThreshold = 100  
    ratio = 3  
    kernel_size = 3  
      
    img = cv2.imread('D:/qq1.jpg')  
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
      
    cv2.namedWindow('canny demo')  
      
    cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)  
      
    CannyThreshold(0)  # initialization  
    if cv2.waitKey(0) == 27:  
        cv2.destroyAllWindows()

def canny_2():
    def nothing(*arg):  
        pass  
      
    cv2.namedWindow('edge')  
    cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)  
    cv2.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)  
  
    cap = cv2.VideoCapture(0)  
    while True:  
        #flag, img = cap.read()
        img = cv2.imread('D:/qq1.jpg')
        img = cv2.GaussianBlur(img, (3,3), 0)   
          
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')  
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')  
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)  
        vis = img.copy()  
        vis = np.uint8(vis/2.)  
        vis[edge != 0] = (0, 255, 0)  
        cv2.imshow('edge', vis)  
        ch = cv2.waitKey(5) & 0xFF  
        if ch == 27:  
            break  
    cv2.destroyAllWindows()  

if __name__ == '__main__':
    canny_1()