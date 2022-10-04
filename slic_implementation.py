from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries,find_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2 as cv
import numpy as np
# construct the argument parser and parse the arguments
img=cv.imread('./countertop_with_markers/RGB_new/rgb18.jpg')

blank=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
blank2=np.zeros(img.shape,dtype='uint8')
blank3=np.zeros(img.shape,dtype='uint8')

# load the image and convert it to a floating point data type
image = img_as_float(img)
# loop over the number of segments
numSegments=2
# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, n_segments = numSegments,compactness=20,sigma=5)
print(np.unique(segments))
for i in range(len(segments)):
	for j in range(len(segments[0])):
		if segments[i][j]==1:
			blank2[i][j]=(0,0,255)
			blank[i][j]=255

contours, hierarchy = cv.findContours(blank, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
max_contour=max(contours,key=cv.contourArea)
hull = cv.convexHull(max_contour)
epsilon = 0.03 * cv.arcLength(hull, True)
	# print(epsilon)
approximations = cv.approxPolyDP(hull, epsilon, True)
cv.drawContours(blank2, [approximations], 0, (0,255,0), 3)
# cv.drawContours(blank2, [hull], 0, (0,255,0), 3)


cv.imshow("BLANK",blank2)
bond=find_boundaries(segments)
print(np.unique(bond))
for i in range(len(bond)):
	for j in range(len(bond)):
		if bond[i][j]==True:
			blank3[i][j]=(255,0,0)
# cv.imshow("NEW",blank3)			
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)

ax.imshow(mark_boundaries(image, segments))
plt.axis("off")
# show the plots
plt.show()

cv.waitKey(0)