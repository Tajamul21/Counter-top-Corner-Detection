import cv2 as cv
import numpy as np
from tqdm import tqdm
import os
import re
def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	'''
	alist.sort(key=natural_keys) sorts in human order
	http://nedbatchelder.com/blog/200712/human_sorting.html
	(See Toothy's implementation in the comments)
	'''
	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def erode_dilate(image):
    """
    Function to perform morphological operations to fill the 'holes' in the threshold image
    """
    kernel_erosion = np.ones((5, 5), np.uint8)
    kernel_dilation=np.ones((5,5),np.uint8)

    img_erosion = cv.erode(image, kernel_erosion, iterations=1)
    img_dilation = cv.dilate(img_erosion, kernel_dilation, iterations=2)

    return img_dilation
images_arr=os.listdir('./dark_film_images/')
images_arr.sort(key=natural_keys)

for i in tqdm(range(len(images_arr))):
	img=cv.imread('./dark_film_images/'+images_arr[i])
	img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
	blank=np.zeros(img.shape,dtype='uint8')
	h,s,v=cv.split(img_hsv)
	mask=cv.inRange(s,120,255)
	for m in range(len(mask)):
		for n in range(len(mask[0])):
			if mask[m][n]==255:
				blank[m][n]=(0,0,255)
	# contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	# max_contour=max(contours,key=cv.contourArea)
	# cv.imshow('Previous',blank)

	blank = cv.bilateralFilter(blank, 5,10,75)
	blank=erode_dilate(blank)
	contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	max_contour=max(contours,key=cv.contourArea)
	print("max_contour.shape",max_contour.shape)


	hull = cv.convexHull(max_contour)
	approximations=[]
	thresh=0.01
	iterations=0
	while len(approximations)!=4 and iterations<20:
		epsilon = thresh * cv.arcLength(hull, True)
		# print(epsilon)
		approximations = cv.approxPolyDP(hull, epsilon, True)
		thresh=thresh+0.005
		iterations+=1


	cv.drawContours(blank, [approximations], 0, (0,255,0), 3)
	for k in range(len(approximations)):
		blank=cv.circle(blank,(approximations[k][0][0],approximations[k][0][1]),10,(7,237,229),-1)
		blank = cv.putText(blank, str(approximations[k][0]), (approximations[k][0][0]+10,approximations[k][0][1]-10), cv.FONT_HERSHEY_SIMPLEX, 
				   0.7, (7,237,229),2, cv.LINE_AA)


		cv.imwrite('./color_segment_counter_results/'+images_arr[i],blank)

cv.waitKey(0)