import pcl
import numpy as np
# import pcl.pcl_visualization
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re

from math import sqrt
def erode_dilate(image):
	"""
	Function to perform morphological operations to fill the 'holes' in the threshold image
	"""
	kernel = np.ones((5, 5), np.uint8)
	img_erosion = cv.erode(image, kernel, iterations=1)
	img_dilation = cv.dilate(img_erosion, kernel, iterations=2)

	return img_dilation
def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	'''
	alist.sort(key=natural_keys) sorts in human order
	http://nedbatchelder.com/blog/200712/human_sorting.html
	(See Toothy's implementation in the comments)
	'''
	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_plane(ptcloud,threshold):
	# model_p = pcl.SampleConsensusModelPlane(ptcloud)
	# ransac = pcl.RandomSampleConsensus (model_p)
	# ransac.set_DistanceThreshold (threshold)
	# ransac.computeModel()
	# inliers = ransac.get_Inliers()

	seg = ptcloud.make_segmenter_normals(ksearch=50)
	seg.set_optimize_coefficients(True)
	seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
	seg.set_method_type(pcl.SAC_RANSAC)
	seg.set_distance_threshold(0.01)
	seg.set_normal_distance_weight(0.01)
	seg.set_max_iterations(100)
	inliers, coefficients = seg.segment()

	return inliers,coefficients

def visualize_cloud(visual_cloud):
	visual = pcl.pcl_visualization.CloudViewing()
	visual.ShowMonochromeCloud(visual_cloud, b'cloud')

	v = True
	while v:
		v = not(visual.WasStopped())

def convert_pointcloud_to_depth(pointcloud):
	"""
	Convert the world coordinate to a 2D image coordinate
	Parameters:
	-----------
	pointcloud       : numpy array with shape 3xN
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x coordinate in image
	y : array
		The y coordiante in image
	"""

	assert (pointcloud.shape[0] == 3)
	x_ = pointcloud[0,:]
	y_ = pointcloud[1,:]
	z_ = pointcloud[2,:]

	m = x_[np.nonzero(z_)]/z_[np.nonzero(z_)]
	n = y_[np.nonzero(z_)]/z_[np.nonzero(z_)]

	x = m*597.404+ 318.665
	y = n*597.404 + 237.591

	return x, y

def convert_to_2dpixel(X3d,Y3d,Z3d):
	x=int((896.106*X3d+637.998*Z3d)/Z3d)
	y=int((896.106*Y3d+356.386*Z3d)/Z3d)
	return x,y


def reproject_points(cloud_arr,plane_inliers,img,coefficients):
	X=[]
	Y=[]
	Z=[]
	img_copy=img.copy()
	blank=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
	blank2=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
	blank3=np.zeros((img.shape[0],img.shape[1],3),dtype='uint8')

	for i in range(len(plane_inliers)):
		X.append(cloud_arr[plane_inliers[i]][0])
		Y.append(cloud_arr[plane_inliers[i]][1])
		Z.append(cloud_arr[plane_inliers[i]][2])

	fig = plt.figure()
	ax = plt.axes(projection='3d')	
	ax.scatter3D(X, Y, Z, c=Z);
	test_x=sum(X)/len(X)
	test_y=sum(Y)/len(Y)
	test_z=sum(Z)/len(Z)


	ax.scatter(test_x,test_y,test_z,c='red',marker='o')
	magnitude=sqrt((coefficients[0]**2)+(coefficients[1]**2)+(coefficients[2]**2)+(coefficients[3]**2))
	arrow_X=test_x+(coefficients[0]/magnitude)
	arrow_Y=test_y+(coefficients[1]/magnitude)
	arrow_Z=test_z+(coefficients[2]/magnitude)

	#Plot the surface normal
	ax.scatter(arrow_X,arrow_Y,arrow_Z,c='red',marker='o',s=25)
	ax.plot([test_x,arrow_X],[test_y,arrow_Y],[test_z,arrow_Z],color = 'g')
	plt.show()
	# equation_plane(p1,p2,p3)
	img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
	h,s,v=cv.split(img_hsv)

	mask=cv.inRange(s,140,255)
	real=[X,Y,Z]
	real=np.array(real)
	# print?(real)
	X_i,Y_i=convert_pointcloud_to_depth(real)
	for j in range(len(X_i)):
		if mask[int(Y_i[j]),int(X_i[j])]==255:
			img_copy[int(Y_i[j]),int(X_i[j])]=(0,0,255)
			blank[int(Y_i[j]),int(X_i[j])]=255
	img_copy=erode_dilate(img_copy)
	contours, hierarchy = cv.findContours(blank, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	max_contour=max(contours,key=cv.contourArea)
	hull = cv.convexHull(max_contour)
	

# Threshold for an optimal value, it may vary depending on the image.
	epsilon = 0.05 * cv.arcLength(hull, True)
	# print(epsilon)
	approximations = cv.approxPolyDP(hull, epsilon, True)
	cv.drawContours(img_copy, [approximations], 0, (0,255,0), 3)

	for k in range(len(approximations)):
		img_copy=cv.circle(img_copy,(approximations[k][0][0],approximations[k][0][1]),10,(7,237,229),-1)
		image_copy = cv.putText(img_copy, str(approximations[k][0]), (approximations[k][0][0]+10,approximations[k][0][1]-10), cv.FONT_HERSHEY_SIMPLEX, 
				   0.7, (7,237,229),2, cv.LINE_AA)


	return img_copy

world_point_data_arr=os.listdir('./dark_film_pointcloud_data/')
images_arr=os.listdir('./dark_film_images/')
world_point_data_arr.sort(key=natural_keys)
images_arr.sort(key=natural_keys)

def convert_to_3D(points, pcl):
	i = 0;
	for i in pcl:
			if np.any(points[0] == pcl[i][0]):
				z =  np.where(points[0] == pcl[i][0])
				return z
			else:
				return points[0], pcl[i][0]
			# if points[0] == pcl[i][j]:
			# 	return i
			# else:
			# 	return points[0], i[i][j]




for ind in tqdm(range(17, 18)):#len(world_point_data_arr))):
	#world_points=pcl.PointCloud()
	world_points_arr=np.load('./dark_film_pointcloud_data/'+world_point_data_arr[ind])
	new = world_points_arr * 1000
	pcl = new.astype(int)
	print(pcl.shape)

	print(pcl)
	p = [202, 322]


	# print(convert_to_3D(p, pcl))
	for i in range(len(pcl)):

		if np.logical_and((pcl[i][0] == p[0]), (pcl[i][1] == p[1])).any():
			print(pcl[i])



	# img_fin=cv.imread('./dark_film_images/'+images_arr[ind])
	# world_points.from_array(world_points_arr)
	# visualize_cloud(world_points)

	# plane1_inliers, plane1_coef = get_plane(world_points,0.01)
	#
	# plane1_img=reproject_points(world_points_arr,plane1_inliers,img_fin,plane1_coef)
	#
	# second_cloud=pcl.PointCloud()
	# second_cloud_arr=np.delete(world_points_arr,plane1_inliers,0)
	# second_cloud.from_array(second_cloud_arr)
	# plane2_inliers,plane2_coef=get_plane(second_cloud,0.01)
	# plane2_img=reproject_points(second_cloud_arr,plane2_inliers,img_fin,plane2_coef)
	#
	# cv.imshow("PLANE 1",plane1_img)
	# cv.imshow("plane 2",plane2_img)

cv.waitKey(0)