from unrealcv import client
import sys
import os
import numpy as np
import cv2
import io

camera_poses=np.array([[-251.007,351.244,161.460,0,273-360.262,0.000],
[-249.598,276.740,163.645,3.186,269.158-360,0.000],
[-250.416,221.061,166.744,3.186,269.158-360,0.000],
[-244.952,177.398,170.926,5.306,277.483-360,0.000],
[-233.936,140.583,175.358,6.488,286.938-360,0.000],
[-212.879,99.808,181.000,7.131,304.636-360,0.000],
[-173.019,58.518,188.487,7.188,313.466-360,0.000],
[-132.732,14.410,194.457,3.946,312.594-360,0.000],
[-101.393,-11.168,197.331,4.065,320.780-360,0.000],
[-45.377,-47.021,200.812,2.996,326.690-360,0.000],
[-12.626,-69.094,202.755,2.755,326.002-360,0.000],
[19.679,-91.812,204.739,2.877,324.788-360,0.000],
[76.785,-119.209,208.925,1.736,334.644-360,0.000],
[105.958,-133.033,209.904,1.736,334.644-360,0.000],
[133.131,-140.153,210.462,1.138,345.318-360,0.000],
[161.093,-142.643,211.611,2.343,354.912-360,0.000],
[188.182,-135.225,212.195,-0.559,25.384,0.000],
[255.793,-96.240,212.415,-2.563,40.913,0.000],
[255.793,-96.240,212.415,-0.459,69.374,0.000],
[255.793,-96.240,212.415,-1.724,82.754,0.000],
[255.793,-96.240,212.415,-1.958,98.535,0.000],
[255.793,-96.240,212.415,-2.518,123.510,0.000],
[216.745,-52.987,211.021,-0.641,136.681,0.000],
[173.676,-16.896,211.314,0.360,140.212,0.000],
[133.707,9.388,211.860,-0.792,148.646,0.000],
[85.765,38.700,211.655,-0.792,147.962,0.000],
[36.397,69.136,206.179,-8.482,148.097,0.000],
[-3.299,95.024,199.443,-8.849,146.905,0.000],
[-35.851,116.812,194.036,-7.217,146.030,0.000],
[-99.200,159.332,183.607,-7.217,146.030,0.000],
[-143.108,197.076,177.431,-3.827,119.731,0.000],
[-164.375,239.956,176.836,-0.287,116.343,0.000],
[-185.483,282.914,176.241,-0.287,116.343,0.000],
[-203.028,318.348,175.749,2.253,108.299,0.000],
[-220.191,371.795,178.018,1.897,106.273,0.000],
[-235.280,425.875,180.287,2.963,104.242,0.000],
[-242.845,473.007,183.549,3.907,96.916,0.000],
[-249.178,528.721,210.091,0.847,91.554,0.000],
[-249.940,556.804,235.481,0.847,91.554,0.000],
[-251.026,607.776,236.261,-1.021,83.742,0.000],
[-244.512,646.766,235.275,-1.633,80.856,0.000],
[-230.276,703.268,234.489,-0.350,75.640,0.000],
[-215.140,748.643,233.209,-4.792,64.573,0.000],
[-184.849,811.243,228.242,-5.715,57.878,0.000],
[-161.436,842.920,224.768,-5.960,53.544,0.000]])

fps = 10
times = np.arange(0,camera_poses.shape[0]*fps,fps)
filled_times = np.arange(0,camera_poses.shape[0]*fps)

filtered_poses = np.array([np.interp(filled_times, times, axis) for axis in camera_poses.T]).T

class UnrealcvStereo():

    def __init__(self):

        client.connect() 
        if not client.isconnected():
            print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
            sys.exit(-1)

    def __str__(self):
        return client.request('vget /unrealcv/status')

    @staticmethod
    def set_position(pose):

        # Set position of the first camera
        client.request(f'vset /camera/0/location {pose[0]} {pose[1]} {pose[2]}')
        client.request(f'vset /camera/0/rotation {pose[3]} {pose[4]} {pose[5]}')

    @staticmethod
    def get_stereo_pair(eye_distance):
        res = client.request('vset /action/eyes_distance %d' % eye_distance)
        res = client.request('vget /camera/0/lit png')
        left = cv2.imdecode(np.frombuffer(res, dtype='uint8'), cv2.IMREAD_UNCHANGED)
        res = client.request('vget /camera/1/lit png')
        right = cv2.imdecode(np.frombuffer(res, dtype='uint8'), cv2.IMREAD_UNCHANGED)

        return left, right

    @staticmethod
    def convert_depth(PointDepth, f=320):
        H = PointDepth.shape[0]
        W = PointDepth.shape[1]
        i_c = np.float(H) / 2 - 1
        j_c = np.float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
        DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
        PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
        return PlaneDepth

    @staticmethod
    def get_depth():

        res = client.request('vget /camera/0/depth npy')
        point_depth = np.load(io.BytesIO(res))

        return UnrealcvStereo.convert_depth(point_depth)


    @staticmethod
    def color_depth(depth_map, max_dist):

        norm_depth_map = 255*(1-depth_map/max_dist)
        norm_depth_map[norm_depth_map < 0] =0
        norm_depth_map[depth_map == 0] =0

        return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)


if __name__ == '__main__':

    eye_distance = 10
    max_depth = 10
    stereo_generator = UnrealcvStereo()

    left_video = cv2.VideoWriter("left_video.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))
    right_video = cv2.VideoWriter("right_video.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))
    depth_folder = "depthmap"

    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)

    for i, pose in enumerate(filtered_poses):

        stereo_generator.set_position(pose)

        # Set the eye distance
        left, right = stereo_generator.get_stereo_pair(eye_distance)

        depth_map = stereo_generator.get_depth()

        depth_map[depth_map>max_depth] = max_depth
        depth_map_u16 = (depth_map*1000).astype(np.uint16)

        color_depth_map = stereo_generator.color_depth(depth_map, max_depth)
        left = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)

        # Save images
        left_video.write(left)

        right_video.write(right)
        cv2.imwrite(f"{depth_folder}/depth_frame{i:03d}.png",depth_map_u16)   

        # Disp
        combined_image = np.hstack((left, right, color_depth_map))
        cv2.imshow("stereo", combined_image)
       
        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
            break

    left_video.release()
    right_video.release()
    cv2.destroyAllWindows()

