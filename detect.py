import sys
import imutils
from skimage import exposure
import numpy as np
import cv2




def rectangleness(hull):
    rect = cv2.boundingRect(hull)
    rectPoints = np.array([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],[rect[0] + rect[2], rect[1] + rect[3]],[rect[0], rect[1] + rect[3]]])
    intersection_area = cv2.intersectConvexConvex(np.array(rectPoints), hull)[0] 
    rect_area = cv2.contourArea(rectPoints)
    rectangleness = intersection_area/rect_area
    return rectangleness

path = sys.argv[1]
img = cv2.imread(path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
edged = cv2.Canny(gray, 200, 255)


(imz, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
plate_candidates = []
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    area = cv2.contourArea(approx)
    if len(approx) == 4:
        hull = cv2.convexHull(approx,returnPoints = True)
        print(rectangleness(hull))
        plate_candidates.append(approx)
        cv2.drawContours(hull, [approx], -1, (0,255,0), 3)
        
cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
cv2.imshow("Original Image",img)
cv2.waitKey()
