#Anil Kumar Koundal, 21-09-2018
#The script reads in an image detects blobs and displays center of blobs.
'''
References:
https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
https://www.learnopencv.com/blob-detection-using-opencv-python-c/
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("E:/noise_img.png")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 150;
params.maxThreshold = 200;

# Filter by Area.
params.filterByArea = False
params.minArea = 50

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.2

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

detector = cv2.SimpleBlobDetector_create(params)
ret,thresh=cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ret2,thresh2 = cv2.threshold(gray_image,127,255,0)
keypoints = detector.detect(thresh)
im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#plt.imshow(thresh,cmap='gray')
#plt.show()

area=[]
im2, contours, hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
	# calculate moments for each contour
	M = cv2.moments(c)
	area.append(cv2.contourArea(c)) 
	# calculate x,y coordinate of center
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
		cX, cY = 0, 0
	cnts=max(c,key=cv2.contourArea)
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	cv2.circle(im_with_keypoints, (cX, cY), 2, (0, 0, 255), -1)
	cv2.circle(im_with_keypoints, extLeft, 2, (0, 0, 255), -1)
	cv2.circle(im_with_keypoints, extRight, 2, (0, 255, 0), -1)
	cv2.circle(im_with_keypoints, extTop, 2, (255, 0, 0), -1)
	cv2.circle(im_with_keypoints, extBot, 2, (255, 255, 0), -1)
	cv2.putText(im_with_keypoints, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
plt.imshow(thresh,cmap='gray')
plt.show()
plt.imshow(thresh2,cmap='gray')
plt.show()
plt.imshow(im_with_keypoints)
plt.show()
print(area)
