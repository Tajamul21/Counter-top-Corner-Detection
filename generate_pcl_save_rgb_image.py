import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
import cv2


def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
    """
	Convert the depthmap to a 3D point cloud
	Parameters:
	-----------
	depth_frame          : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters
	"""

    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    z = depth_image.flatten() / 1000;
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return x, y, z


def convert_pointcloud_to_depth(pointcloud, camera_intrinsics):
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

    assert (pointcloud.shape[1] == 3)
    x_ = pointcloud[0, :]
    y_ = pointcloud[1, :]
    z_ = pointcloud[2, :]

    m = x_[np.nonzero(z_)] / z_[np.nonzero(z_)]
    n = y_[np.nonzero(z_)] / z_[np.nonzero(z_)]

    x = m * camera_intrinsics.fx + camera_intrinsics.ppx
    y = n * camera_intrinsics.fy + camera_intrinsics.ppy

    return x, y


def get_depth_at_pixel(depth_frame, pixel_x, pixel_y):
    """
	Get the depth value at the desired image point
	Parameters:
	-----------
	depth_frame      : rs.frame()
						   The depth frame containing the depth information of the image coordinate
	pixel_x              : double
						   The x value of the image coordinate
	pixel_y              : double
							The y value of the image coordinate
	Return:
	----------
	depth value at the desired pixel
	"""
    return [pixel_x, pixel_y, depth_frame.as_depth_frame().get_distance(round(pixel_y), round(pixel_x))]


def convert_to_camera_cord(u, v, d, intrin):
    x_over_z = (intrin.ppx - u) / intrin.fx
    y_over_z = (intrin.ppy - v) / intrin.fy
    z = d / np.sqrt(1. + x_over_z ** 2 + y_over_z ** 2)
    x = -1 * (x_over_z * z)
    y = -1 * (y_over_z * z)
    # x=round(x,2)
    # y=round(y,2)
    # z=round(z,2)
    return x, y, z


config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
align_to = rs.stream.color
align = rs.align(align_to)
pipeline = rs.pipeline()
pipe_profile = pipeline.start(config)
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
extrinsics = color_frame.profile.get_extrinsics_to(depth_frame.profile)
depth_sensor = pipe_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(depth_intrin)


# Streaming loop
count = 0
while True:  # Get frame set of color and depth
    frames = pipeline.wait_for_frames()

    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())

    # print(depth_image.shape)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("IMG", color_image)
    cv2.imshow("IMGd", depth_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('s'):

        count += 1
        Depth = []
        threed = []
        X, Y, Z = convert_depth_frame_to_pointcloud(depth_image, depth_intrin)
        for i in range(len(X)):
            threed.append([X[i], Y[i], Z[i]])
        threed = np.array(threed)
        threed = np.float32(threed)

        np.save('T:\ARTPARK\Work\Artpark-work-progress\JNTATA_Environment_testing\Countertop\pcl' + str(count), threed)
        cv2.imwrite('T:\ARTPARK\Work\Artpark-work-progress\JNTATA_Environment_testing\Countertop\Color' + str(count) + '.jpg', color_image)
        cv2.imwrite('T:\ARTPARK\Work\Artpark-work-progress\JNTATA_Environment_testing\Countertop\Depth' + str(count) + '.jpg', depth_image)
        # cv2.imshow("img", color_frame)

        print('Image ' + str(count) + ' saved')

pipeline.stop()
