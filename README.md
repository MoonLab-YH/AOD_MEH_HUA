# AL_ObjectDetection_MEH_HUA
Official Pytorch implementation for the paper titled "Active Learning for Object Detection with Evidential Deep Learning and Hierarchical Uncertainty Aggregation" presented on ICLR 2023.


Obtaining concentration parameter $\alpha$             |  Calculation of epistemic uncertainty with sampling
:-------------------------:|:-------------------------:
![Fig_Img2Dir](https://user-images.githubusercontent.com/54431060/221471465-90994e7d-7bf6-43b1-91c0-0af11631de7a.jpg)   |  ![Fig_Dir2Cat3](https://user-images.githubusercontent.com/54431060/221471621-0dc67520-92ac-41f7-a069-6e6f57eb833f.jpg)

# Abstract
Despite the huge success of object detection, the training process still requires an immense amount of labeled data. Although various active learning solutions for object detection have been proposed, most existing works do not take advantage of epistemic uncertainty, which is an important metric for capturing the usefulness of the sample. Also, previous works pay little attention to the attributes of each bounding box (e.g., nearest object, box size) when computing the informativeness of an image. In this paper, we propose a new active learning strategy for object detection that overcomes the shortcomings of prior works. To make use of epistemic uncertainty, we adopt evidential deep learning (EDL) and propose a new module termed model evidence head (MEH), that makes EDL highly compatible with object detection. Based on the computed epistemic uncertainty of each bounding box, we propose hierarchical uncertainty aggregation (HUA) for obtaining the informativeness of an image. HUA realigns all bounding boxes into multiple levels based on the attributes and aggregates uncertainties in a bottom-up order, to effectively capture the context within the image. Experimental results show that our method outperforms existing state-of-the-art methods by a considerable margin.

# Environment Info
```
sys.platform: linux

Python: 3.7.10 (default, Jun  4 2021, 14:48:32) [GCC 7.5.0]  
Pytorch : 1.5.0  
TorchVision: 0.6.0  
Cudatoolkit : 10.1.243  
OpenCV: 4.5.2  
MMCV: 1.3.8  
MMDetection: 2.13.0  
MMDetection Compiler: GCC 7.3  
MMDetection CUDA Compiler: 10.1  
```

# Running Code
```
#For RetinaNet 

python tools/train_RetinaNet.py     --gpu-ids {GPU device number}
                                    --work_dir {dir to save logs and models}
                                    --config {train config file path}                             
                                    
#For SSD

python tools/train_SSD.py           --gpu-ids {GPU device number}
                                    --work_dir {dir to save logs and models}
                                    --config {train config file path}     
```

# Acknowledgement
Our code is based on the implementations of [Multiple Instance Active Learning for Object Detection](https://github.com/yuantn/MI-AOD).
