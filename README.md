# TFLite-msg_chn_wacv20-depth-completion
Python script for performing depth completion from sparse depth and rgb images using the msg_chn_wacv20. model in Tensorflow Lite.

![TFLite msg_chn_wacv20 depth completion](https://github.com/ibaiGorordo/TFLite-msg_chn_wacv20-depth-completion/blob/main/doc/img/out.png)

# Requirements

 * **OpenCV** and **tensorflow or tflite_runtime**. Also, **unrealcv** is only required if you want to generate new data using unrealcv.

For the tflite runtime, you can either use tensorflow `pip install tensorflow` or the [TensorFlow Runtime binary](https://github.com/PINTO0309/TensorflowLite-bin)

# UnrealCV synthethic data generation
The input images and depth are generated using the UnrealCV library (https://unrealcv.org/), you can find more information about how to generate this data in this [other repository for Unreal Synthetic depth generation](https://github.com/ibaiGorordo/UnrealCV-stereo-depth-generation).

# Installation
```
pip install -r requirements.txt
```

# TFLite model
The original models were converted to different formats (including .tflite) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/160_msg_chn_wacv20) and save them into the **[models](https://github.com/ibaiGorordo/TFLite-msg_chn_wacv20-depth-completion/tree/main/models)** folder. 

# Original Pytorch model
The Pytorch pretrained model was taken from the [original repository](https://github.com/anglixjtu/msg_chn_wacv20).

# ONNX inference
For ONNX inference, use this other repository:https://github.com/ibaiGorordo/ONNX-msg_chn_wacv20-depth-completion
 
# Examples

  * **Video inference (UnrealCV synthetic data)**:
 
 ```
 python videoDepthEstimation.py
 ```
 
# Inference video Example
![TFLite msg_chn_wacv20 depth completion](https://github.com/ibaiGorordo/TFLite-msg_chn_wacv20-depth-completion/blob/main/doc/img/msg_chn_wacv20%20depth%20completion3.gif)

# References:
* msg_chn_wacv20 model: https://github.com/anglixjtu/msg_chn_wacv20
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow

* Original paper: 
https://openaccess.thecvf.com/content_WACV_2020/papers/Li_A_Multi-Scale_Guided_Cascade_Hourglass_Network_for_Depth_Completion_WACV_2020_paper.pdf


