The robot needed to find the extreme edges of the counter-top and the sink. To detect
the corners, we used three methods to obtain the coordinates of 8 corners (4-counter top
and 4-sink) as shown in figure. The three methods were:
1. Finding corner points using P4P(4 point algorithm)
2. Finding corner points using RANSAC.
3. Finding corner points using colour segmentation.
They are explained below.
4.1 Method 1: Finding corner points using P4P
The counter top had round colour lables stuck on them in the four outer and inner
corners. (See Figure 13 for an example.) The outer points A, B, C , D in figure 13 were
segmented based on color and the coordinates were saved. Using 4 Point algorithm, we
were able to re project the outer and inner corners of the counter-top with an average
re-projection error of 0.489. We can also compute the 3D locations of these points with
respect to the camera coordinate frame.

4.2 Method 2: Finding corner points using RANSAC
RANSAC divides data into inliers and outliers and yields an estimate of the counter
top plane, computed from a minimal set of inliers with greatest support. We Improved
this initial estimate with a Least Squares (S) estimation over all inliers (i.e., standard
minimization), and then we found the inliers w.r.t the L.S. fit, and re-estimated the plane
using L.S. one more time. We used the 3D points (given by the camera) to find the
counter plane using this method. We then identified the pixels corresponding to the
plane in the corresponding colour image. To find the corners, we used Harris corner
detector. Harris’ corner detector takes the differential of the corner score into account
with reference to direction directly, instead of using shifting patches for every 45◦ angles,
and has been proved to be more accurate in distinguishing between edges and corners.
The implementation was done as follows:
1. Compute image intensity gradients in x- and y-directions.
2. Blur output of (1)
3. Computed Harris response over output of (2)
4. Suppress non-maximas in output of (3) in a 3×3-neighborhood and threshold output

Once the corner points were found, the 3D points can be obtained by using the corre-
sponding points in the depth image.

4.3 Method 3: Finding corner points using color segmentation
The counter top is segmented based on color and the 4 corners can be obtained based
on the fitting of a convex hull. Segmentation is the process of dividing an image into its

constituent parts or objects. Common techniques include edge detection, boundary de-
tection, thresholding, region based segmentation, among others. For this task, we focused

on segmenting our images using Color Image Segmentation through the HSV color space.
Once the corner points were found by using this method, the 3D points can be obtained by using the corresponding points in the depth image.


