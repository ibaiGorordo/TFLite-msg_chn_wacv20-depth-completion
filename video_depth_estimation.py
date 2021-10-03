import cv2
import os
import numpy as np

from msg_chn_wacv20 import Msg_chn_wacv20
from msg_chn_wacv20.utils import make_depth_sparse, draw_depth, update_depth_density

if __name__ == '__main__':
    
    depth_density = 10
    depth_density_rate = -0.2
    max_depth = 10

    model_path='models/saved_model_480x640/model_float32.tflite'
    depth_estimator = Msg_chn_wacv20(model_path)

    cap_depth = cv2.VideoCapture("indoor_example/depthmap/depth_frame%03d.png", cv2.CAP_IMAGES)
    cap_rgb = cv2.VideoCapture("indoor_example/left_video.avi")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    i = 0
    while cap_rgb.isOpened() and cap_depth.isOpened():

        # Read frame from the videos
        ret, rgb_frame = cap_rgb.read()

        ret, depth_frame = cap_depth.read()

        if not ret:
            break

        depth_frame = depth_frame/1000 # to m

        # Make the depth map sparse
        depth_density, depth_density_rate = update_depth_density(depth_density, depth_density_rate, 1, 20)
        sparse_depth = make_depth_sparse(depth_frame, depth_density)

        # Fill the sparse depth map
        estimated_depth = depth_estimator(rgb_frame, sparse_depth)

        # Color depth maps
        color_gt_depth = draw_depth(depth_frame, max_depth)
        color_sparse_depth = draw_depth(sparse_depth, max_depth)
        color_estimated_depth = draw_depth(estimated_depth, max_depth)
    
        combined_img = np.vstack((np.hstack((rgb_frame, color_sparse_depth)),np.hstack((color_gt_depth,color_estimated_depth))))
        cv2.putText(combined_img,f'Density:{depth_density:.1f}%',(combined_img.shape[1]-320,rgb_frame.shape[0]+50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
        cv2.imshow("Estimated depth", combined_img)
        cv2.waitKey(1)