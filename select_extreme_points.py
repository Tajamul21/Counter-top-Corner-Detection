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
	x_cords=[]
	y_cords=[]
	for point in max_contour:
		x_cords.append(point[0][0])
		y_cords.append(point[0][1])
	x_min=min(x_cords)
	x_min_ind=x_cords.index(x_min)
	p1=(x_min,y_cords[x_min_ind])

	y_min=min(y_cords)
	y_min_ind=y_cords.index(y_min)
	p2=(x_cords[y_min_ind],y_min)

	x_max=max(x_cords)
	x_max_ind=x_cords.index(x_max)
	p3=(x_max,y_cords[x_max_ind])

	y_max=max(y_cords)
	y_max_ind=y_cords.index(y_max)
	p4=(x_cords[y_max_ind],y_max)
	print(p1,' ',p2,' ',p3,' ',p4)
	blank=cv.circle(blank,p1,10,(0,255,0),-1)
	blank=cv.circle(blank,p2,10,(0,255,0),-1)
	blank=cv.circle(blank,p3,10,(0,255,0),-1)
	blank=cv.circle(blank,p4,10,(0,255,0),-1)

	cv2.imshow("img", blank)
	# cv.imwrite('./extreme_points/'+images_arr[i],blank)

cv.waitKey(0)