import sys
import cv2
import time
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

class Msg_chn_wacv20():

    def __init__(self, model_path):

        # Initialize model
        self.model = self.initialize_model(model_path)

    def __call__(self, image, sparse_depth, max_depth = 10.0):

        return self.estimate_depth(image, sparse_depth, max_depth)

    def initialize_model(self, model_path):

        self.interpreter = Interpreter(model_path=model_path, num_threads = 4)
        self.interpreter.allocate_tensors()

        # Get model info
        self.getModel_input_details()
        self.getModel_output_details()

    def estimate_depth(self, image, sparse_depth, max_depth = 10.0):
        
        self.max_depth = max_depth

        input_rgb_tensor = self.prepare_rgb(image)
        input_depth_tensor = self.prepare_depth(sparse_depth)

        outputs = self.inference(input_rgb_tensor, input_depth_tensor)
        
        depth_map = self.process_output(outputs)

        return depth_map

    def prepare_rgb(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.img_height, self.img_width, self.img_channels = img.shape

        img_input = cv2.resize(img, (self.input_width,self.input_height))
        img_input = img_input/255
        img_input = img_input[np.newaxis,:,:,:]        

        return img_input.astype(np.float32)

    def prepare_depth(self, img):

        img_input = cv2.resize(img, (self.input_width,self.input_height), interpolation = cv2.INTER_NEAREST)

        img_input = img_input/self.max_depth*256 # 0-256
        img_input = img_input[np.newaxis,:,:,np.newaxis]        

        return img_input.astype(np.float32)


    def inference(self, input_rgb_tensor, input_depth_tensor):
        start = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_rgb_tensor)
        self.interpreter.set_tensor(self.input_details[1]['index'], input_depth_tensor)
        self.interpreter.invoke()

        outputs = [self.get_output_tensor(i) for i in range(self.output_len)]

        return outputs

    def get_output_tensor(self, index):

        tensor = np.squeeze(self.interpreter.get_tensor(self.output_details[index]['index']))
        return tensor

    def process_output(self, outputs):  

        depth_map = np.squeeze(outputs[0])/256 # 0-256 -> 0-1
        depth_map = depth_map*self.max_depth # 0-max_depth

        return cv2.resize(depth_map, (self.img_width, self.img_height)) 

    def getModel_input_details(self):

        self.input_details = self.interpreter.get_input_details()

        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.channels = input_shape[3]

    def getModel_output_details(self):

        self.output_details = self.interpreter.get_output_details()
        self.output_len = len(self.output_details)



if __name__ == '__main__':

    from utils import make_depth_sparse, draw_depth, update_depth_density

    depth_density = 10
    depth_density_rate = 0.2
    max_depth = 10

    model_path='../models/saved_model_480x640/model_float32.tflite'
    depth_estimator = Msg_chn_wacv20(model_path)

    cap_depth = cv2.VideoCapture("../indoor_example/depthmap/depth_frame%03d.png", cv2.CAP_IMAGES)
    cap_rgb = cv2.VideoCapture("../indoor_example/left_video.avi")

    while cap_rgb.isOpened() and cap_depth.isOpened():

        # Read frame from the videos
        ret, rgb_frame = cap_rgb.read()

        ret, depth_frame = cap_depth.read()

        if not ret:
            break

        depth_frame = depth_frame/1000 # to m

        # Make the depth map sparse
        depth_density, depth_density_rate = update_depth_density(depth_density, depth_density_rate, 1, 10)
        sparse_depth = make_depth_sparse(depth_frame, depth_density)

        # Fill the sparse depth map
        estimated_depth = depth_estimator(rgb_frame, sparse_depth)

        # Color depth maps
        color_gt_depth = draw_depth(depth_frame, max_depth)
        color_sparse_depth = draw_depth(sparse_depth, max_depth)
        color_estimated_depth = draw_depth(estimated_depth, max_depth)
    
        combined_img = np.vstack((np.hstack((rgb_frame, color_sparse_depth)),np.hstack((color_gt_depth,color_estimated_depth))))
        cv2.putText(combined_img,f'Density:{depth_density:.1f}%',(combined_img.shape[1]-500,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
       
        cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
        cv2.imshow("Estimated depth", combined_img)
        cv2.waitKey(1)
