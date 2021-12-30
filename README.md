# DEKRv2: More Fast or Accurate than DEKR

## Introduction
Bottom-up human pose estimation has raised more investigation in recent years, especially dense keypoint regression. However, the state-of-art DEKR still has some aspects (e.g. speed and accuracy) to be improved. In this paper, we propose a new framework named **DEKRv2**, which has been improved compared to DEKR. We find that the multi-branch network in DEKR is very time-consuming because it is serial. We designed a more effective module based on Group Convolution to replace multi-branches network in DEKR, it can reduce reasoning time. When DEKR calculates the offset of each keypoint, it only considers the features of the current keypoint, and neglects the constraints between the adjacent keypoints. we adopt a coarse-to-fine feature extraction method to obtain more accurate feature location of keypoints for this problem. 
In addition, we conclude that it makes no sense that modifying adaptive convolution. 
Experiments on the CrowdPose dataset show that our  method achieves superior compared with DEKR in speed or accuracy, respectively. 
		
## Main Results
### Results on CrowdPose test without multi-scale test with rescorenet 300 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18**           | 0.660 | 0.861 | 0.712 | 0.738 | 0.922 | 0.785 | 0.735 | 0.669 | 0.571 |
| **pose_hrnet_w32(paper)**    | 0.657 | 0.857 | 0.704 | 0.723 | 0.906 | 0.769 | 0.730 | 0.664 | 0.574 |
| **pose_hrnet_w18_dc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**        | 0.629 | 0.844 | 0.681 | 0.716 | 0.917 | 0.766 | 0.701 | 0.639 | 0.539 |
| **pose_hrnet_w18_gc2**       | 0.633 | 0.851 | 0.685 | 0.711 | 0.914 | 0.761 | 0.708 | 0.641 | 0.543 |
| **pose_hrnet_w18_gc3**       | 0.659 | 0.863 | 0.709 | 0.736 | 0.923 | 0.781 | 0.732 | 0.667 | 0.570 |
| **pose_hrnet_w18_part**      | 0.654 | 0.856 | 0.705 | 0.735 | 0.921 | 0.782 | 0.728 | 0.662 | 0.562 |
| **pose_hrnet_w18_part2**     | **0.666** | 0.863 | 0.715 | 0.742 | 0.924 | 0.789 | 0.739 | 0.673 | 0.579 |
| **pose_hrnet_w18_part3**     | 0.659 | 0.862 | 0.710 | 0.737 | 0.922 | 0.783 | 0.735 | 0.667 | 0.571 |

### Results on CrowdPose test without multi-scale test no rescorenet 300 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)** | 0.648 | 0.850 | 0.701 | 0.738 | 0.922 | 0.784 | 0.720 | 0.658 | 0.558 |
| **pose_hrnet_w18_dc**        | 0.649 | 0.852 | 0.700 | 0.738 | 0.921 | 0.784 | 0.721 | 0.658 | 0.559 |
| **pose_hrnet_w18_dc_3**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**        | 0.617 | 0.834 | 0.669 | 0.715 | 0.917 | 0.765 | 0.687 | 0.627 | 0.525 |
| **pose_hrnet_w18_gc2**       | 0.619 | 0.840 | 0.672 | 0.711 | 0.914 | 0.761 | 0.694 | 0.628 | 0.527 |
| **pose_hrnet_w18_gc3**       | 0.647 | 0.853 | 0.697 | 0.736 | 0.923 | 0.781 | 0.718 | 0.656 | 0.556 |
| **pose_hrnet_w18_part**      | 0.640 | 0.845 | 0.692 | 0.735 | 0.920 | 0.782 | 0.713 | 0.649 | 0.546 |
| **pose_hrnet_w18_part2**     | 0.654 | 0.854 | 0.704 | 0.742 | 0.923 | 0.789 | 0.727 | 0.662 | 0.566 |
| **pose_hrnet_w18_part3**     | 0.647 | 0.853 | 0.699 | 0.737 | 0.922 | 0.783 | 0.720 | 0.656 | 0.559 |

### Results on CrowdPose test with multi-scale test no rescorenet 300 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)** | 0.659 | 0.830 | 0.714 | 0.753 | 0.919 | 0.803 | 0.745 | 0.670 | 0.549 |
| **pose_hrnet_w18_dc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc2**       | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc3**       | 0.659 | 0.834 | 0.711 | 0.751 | 0.920 | 0.799 | 0.748 | 0.672 | 0.545 |
| **pose_hrnet_w18_part**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_part2**     | 0.661 | 0.833 | 0.716 | 0.754 | 0.919 | 0.803 | 0.748 | 0.672 | 0.552 |
| **pose_hrnet_w18_part3**     | - | - | - | - | - | - | - | - |

### Results on CrowdPose test with multi-scale test with rescorenet 300 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)** | 0.674 | 0.851 | 0.729 | 0.754 | 0.921 | 0.804 | 0.761 | 0.685 | 0.566 |
| **pose_hrnet_w18_dc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**        | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc2**       | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc3**       | 0.674 | 0.852 | 0.727 | 0.753 | 0.922 | 0.801 | 0.762 | 0.686 | 0.561 |
| **pose_hrnet_w18_part**      | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_part2**     | 0.676 | 0.852 | 0.730 | 0.756 | 0.921 | 0.805 | 0.762 | 0.686 | 0.570 |

### Results on CrowdPose test without multi-scale test no rescorenet 100 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)**  | 0.583 | 0.801 | 0.630 | 0.710 | 0.915 | 0.756 | 0.653 | 0.596 | 0.483 |
| **pose_hrnet_w18_dc**         | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**       | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**         | 0.558 | 0.800 | 0.604 | 0.679 | 0.906 | 0.728 | 0.625 | 0.570 | 0.464 |
| **pose_hrnet_w18_gc2**        | 0.562 | 0.809 | 0.606 | 0.673 | 0.904 | 0.719 | 0.636 | 0.572 | 0.469 |
| **pose_hrnet_w18_gc3**        | 0.602 | 0.822 | 0.647 | 0.703 | 0.910 | 0.747 | 0.678 | 0.613 | 0.503 |
| **pose_hrnet_w18_part**       | 0.585 | 0.808 | 0.632 | 0.703 | 0.913 | 0.750 | 0.657 | 0.597 | 0.484 |
| **pose_hrnet_w18_part2**      | 0.586 | 0.805 | 0.633 | 0.710 | 0.914 | 0.756 | 0.656 | 0.599 | 0.490 |
| **pose_hrnet_w18_part_final** | 0.588 | 0.817 | 0.633 | 0.703 | 0.912 | 0.748 | 0.664 | 0.598 | 0.492 |

### Results on CrowdPose test without multi-scale test with rescorenet 100 epoch
| Method             | AP | Ap .5 | AP .75 | AR | AR .5 | AR .75 | AP (easy) | AP (medium) | AP (hard) |
|--------------------|---|---|---|---|---|---|---|---|---|
| **pose_hrnet_w18(baseline)**  | 0.602 | 0.818 | 0.648 | 0.709 | 0.915 | 0.755 | 0.674 | 0.614 | 0.503 |
| **pose_hrnet_w18_dc**         | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_dc_3**       | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc**         | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc2**        | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_gc3**        | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_part**       | - | - | - | - | - | - | - | - | - |
| **pose_hrnet_w18_part2**      | 0.606 | 0.823 | 0.652 | 0.710 | 0.914 | 0.756 | 0.679 | 0.617 | 0.510 |
| **pose_hrnet_w18_part_final** | - | - | - | - | - | - | - | - | - |

### Parms and Flops
| Method             | time |  GFlops | Parms(M) |
|--------------------|------|---------|-------|
|**baseline**        | 175  | 18.806  | 9.655 |
|**gc1**             | 170  | 18.803  | 9.654 |
|**gc2**             | 146  | 18.803  | 9.654 |
|**gc3**             | 174  | 18.806  | 9.655 |


### Note:
- Flip test is used.
- GFLOPs is for convolution and linear layers only.


## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA V100 GPU cards for HRNet-w32 and 8 NVIDIA V100 GPU cards for HRNet-w48. Other platforms are not fully tested.

## Quick start
### Installation
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly the same as COCOAPI.  
5. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── model
   ├── experiments
   ├── lib
   ├── tools 
   ├── log
   ├── output
   ├── README.md
   ├── requirements.txt
   └── setup.py
   ```

6. Download pretrained models and our well-trained models from zoo([OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EmoNwNpq4L1FgUsC9KbWezABSotd3BGOlcWCdkBi91l50g?e=HWuluh)) and make models directory look like this:
    ```
    ${POSE_ROOT}
    |-- model
    `-- |-- imagenet
        |   |-- hrnet_w32-36af842e.pth
        |   `-- hrnetv2_w48_imagenet_pretrained.pth
        |-- pose_coco
        |   |-- pose_dekr_hrnetw32_coco.pth
        |   `-- pose_dekr_hrnetw48_coco.pth
        |-- pose_crowdpose
        |   |-- pose_dekr_hrnetw32_crowdpose.pth
        |   `-- pose_dekr_hrnetw48_crowdpose.pth
        `-- rescore
            |-- final_rescore_coco_kpt.pth
            `-- final_rescore_crowd_pose_kpt.pth
    ```
   
### Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. 
Download and extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- person_keypoints_train2017.json
            |   `-- person_keypoints_val2017.json
            `-- images
                |-- train2017.zip
                `-- val2017.zip

**For CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training.
Download and extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- |-- crowdpose
        `-- |-- json
            |   |-- crowdpose_train.json
            |   |-- crowdpose_val.json
            |   |-- crowdpose_trainval.json (generated by tools/crowdpose_concat_train_val.py)
            |   `-- crowdpose_test.json
            `-- images.zip

After downloading data, run `python tools/crowdpose_concat_train_val.py` under `${POSE_ROOT}` to create trainval set.


### Training and Testing

#### Testing on COCO val2017 dataset without multi-scale test using well-trained pose model

```
CUDA_VISIBLE_DEVICES=1
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_dekr_hrnetw32_coco.pth
```

#### Testing on COCO test-dev2017 dataset without multi-scale test using well-trained pose model

```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_dekr_hrnetw32_coco.pth \ 
    DATASET.TEST test-dev2017
```

#### Testing on COCO val2017 dataset with multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_dekr_hrnetw32_coco.pth \ 
    TEST.NMS_THRE 0.15 \
    TEST.SCALE_FACTOR 0.5,1,2
```

#### Testing on COCO val2017 dataset with matching regression results to the closest keypoints detected from the keypoint heatmaps

```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_dekr_hrnetw32_coco.pth \ 
    TEST.MATCH_HMP True
```

#### Testing on crowdpose test dataset without multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE models/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth
```

#### Testing on crowdpose test dataset with multi-scale test using well-trained pose model
 
```
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE models/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth \ 
    TEST.NMS_THRE 0.15 \
    TEST.SCALE_FACTOR 0.5,1,2
```

#### Testing on crowdpose test dataset with matching regression results to the closest keypoints detected from the keypoint heatmaps
 
```
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE models/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth \ 
    TEST.MATCH_HMP True
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
```

#### Training on Crowdpose trainval dataset

```
python tools/train.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
```

#### Using inference demo
```
python tools/inference_demo.py --cfg experiments/coco/inference_demo_coco.yaml \
    --videoFile ../multi_people.mp4 \
    --outputDir output \
    --visthre 0.3 \
    TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32.pth
python tools/inference_demo.py --cfg experiments/crowdpose/inference_demo_crowdpose.yaml \
    --videoFile ../multi_people.mp4 \
    --outputDir output \
    --visthre 0.3 \
    TEST.MODEL_FILE model/pose_crowdpose/pose_dekr_hrnetw32.pth \
```

The above command will create a video under *output* directory and a lot of pose image under *output/pose* directory. 

#### Scoring net
We use a scoring net, consisting of two fully-connected layers (each followed by a ReLU layer), and a linear prediction layer which aims to learn the
OKS score for the corresponding predicted pose. For this scoring net, you can directly use our well-trained model in the model/rescore folder. You can also train your scoring net using your pose estimation model by the following steps:

1. Generate scoring dataset on train dataset:
```
python tools/valid.py \
    --cfg experiments/coco/rescore_coco.yaml \
    TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32.pth
python tools/valid.py \
    --cfg experiments/crowdpose/rescore_crowdpose.yaml \
    TEST.MODEL_FILE model/pose_crowdpose/pose_dekr_hrnetw32.pth \
```

2. Train the scoring net using the scoring dataset:
```
python tools/train_scorenet.py \
    --cfg experiment/coco/rescore_coco.yaml
python tools/train_scorenet.py \
    --cfg experiments/crowdpose/rescore_crowdpose.yaml \
```

3. Using the well-trained scoring net to improve the performance of your pose estimation model (above 0.6AP).
```
python tools/valid.py \
    --cfg experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml \
    TEST.MODEL_FILE models/pose_coco/pose_dekr_hrnetw32_coco.pth
python tools/valid.py \
    --cfg experiments/crowdpose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_crowdpose_x300.yaml \
    TEST.MODEL_FILE models/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth \
```

### Acknowledge
Our code is mainly based on [DEKR](https://github.com/HRNet/DEKR). 

### Citation

```
@inproceedings{GengSXZW21,
  title={Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression},
  author={Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, Jingdong Wang},
  booktitle={CVPR},
  year={2021}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI}
  year={2019}
}
```


