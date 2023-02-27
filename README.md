# AL_ObjectDetection_MEH_HUA
Official Pytorch implementation for the paper titled "Active Learning for Object Detection with Evidential Deep Learning and Hierarchical Uncertainty Aggregation" presented on ICLR 2023.


Obtaining concentration parameter $\alpha$             |  Calculation of epistemic uncertainty with sampling
:-------------------------:|:-------------------------:
![Fig_Img2Dir](https://user-images.githubusercontent.com/54431060/221471465-90994e7d-7bf6-43b1-91c0-0af11631de7a.jpg)   |  ![Fig_Dir2Cat3](https://user-images.githubusercontent.com/54431060/221471621-0dc67520-92ac-41f7-a069-6e6f57eb833f.jpg)

# Abstract
Despite the huge success of object detection, the training process still requires an immense amount of labeled data. Although various active learning solutions for object detection have been proposed, most existing works do not take advantage of epistemic uncertainty, which is an important metric for capturing the usefulness of the sample. Also, previous works pay little attention to the attributes of each bounding box (e.g., nearest object, box size) when computing the informativeness of an image. In this paper, we propose a new active learning strategy for object detection that overcomes the shortcomings of prior works. To make use of epistemic uncertainty, we adopt evidential deep learning (EDL) and propose a new module termed model evidence head (MEH), that makes EDL highly compatible with object detection. Based on the computed epistemic uncertainty of each bounding box, we propose hierarchical uncertainty aggregation (HUA) for obtaining the informativeness of an image. HUA realigns all bounding boxes into multiple levels based on the attributes and aggregates uncertainties in a bottom-up order, to effectively capture the context within the image. Experimental results show that our method outperforms existing state-of-the-art methods by a considerable margin.

# Running Code
```
#For miniImageNet 

cd /miniImageNet/[miniImageNet][IID] or /miniImageNet/[miniImageNet][Non-IID]
python miniimagenet_train.py        --gpu {GPU device number}
                                    --update_lr {task-lever inner update learning rate}
                                    --meta-lr {meta-level outer learning learning rate}
                                    --round {number of communication round}
                                    --n_user {number of clients}                                  
                                    
#For CIFAR-100

cd /CIFAR100/[CIFAR100][IID] or /CIFAR100/[CIFAR100][Non-IID]
python cifar100_train.py            --gpu {GPU device number}
                                    --update_lr {task-lever inner update learning rate}
                                    --meta-lr {meta-level outer learning learning rate}
                                    --round {number of communication round}
                                    --n_user {number of clients}
```

# Acknowledgement
Our code is based on the implementations of [Multiple Instance Active Learning for Object Detection]([url](https://github.com/yuantn/MI-AOD)).
