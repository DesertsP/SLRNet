# Learning Self-Supervised Low-Rank Network for Single-Stage Weakly and Semi-Supervised Semantic Segmentation
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official implementaion of [Learning Self-Supervised Low-Rank Network for Single-Stage
Weakly and Semi-Supervised Semantic Segmentation](http://dx.doi.org/10.1007/s11263-022-01590-z), [arXiv](https://arxiv.org/abs/2203.10278), IJCV 2022.

This repository contains the code for SLRNet, which is a unified framework that can be well generalized to learn a label-efficient segmentation model in various weakly and semi-supervised settings.
The key component of our approach is the Cross-View Low-Rank (CVLR) module that decompose the multi-view representations via the collective matrix factorization.
We provide scripts for Pascal VOC, COCO and L2ID datasets.

## Setup
0. **Minimum requirements.** 
   This project was developed with Python 3.7, PyTorch 1.x. 
   The training requires at least two Titan XP GPUs (12 GB memory each).
1. **Setup your Python environment.**
   Install python dependencies in requirements.txt
    ```
    pip install -r requirements.txt
    ```
2. **Download Datasets.** 
   
   **Download Pascal VOC data from:**
    - VOC: [Training/Validation (2GB .tar file)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
    - SBD: [Training (1.4GB .tgz file)](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
    
    Convert SBD data using `tools/convert_coco.py`.
   
    Link to the data:
    ```
    ln -s <your_path_to_voc> <project>/data/voc
    ln -s <your_path_to_sbd> <project>/data/sbd
    ```
    Make sure that the first directory in `data/voc` is `VOCdevkit`; the first directory in `data/sbd` is `benchmark_RELEASE`.
   
   **Download COCO data from:**
    - COCO: [Training](http://images.cocodataset.org/zips/train2014.zip), [Validation](http://images.cocodataset.org/zips/val2014.zip), [Annotation](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
    
    Convert COCO data using `tools/convert_sbd.py`.

    **Download L2ID challenge data from:**
    - [Learning from Limited and Imperfect Data (L2ID)](https://l2id.github.io/)

3. **Download pre-trained backbones.** 
   
    | Backbone | Initial Weights | Comment |
    |:---:|:---:|:---:|
    | WideResNet38 | [ilsvrc-cls_rna-a1_cls1000_ep-0001.pth (402M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth) | Converted from [mxnet](https://github.com/itijyou/ademxapp) |
    | ResNet101 | [resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | PyTorch official |

## Pre-trained Model
For testing, we provide our checkpoints on Pascal VOC dataset:

| Setting | Backbone | Val | Link |
| :------- |:---:|:---:|:---:|
| Weakly-sup. w/ image-level label | WideResNet38 | 67.2 (w/ CRF) |  [link](https://drive.google.com/file/d/1jdBYwcbHFgU6l9KYcVEllIdlekenAWmq/view?usp=sharing) |
| Weakly-sup. w/ image-level label (Two-stage) | ResNet101 | 69.3 (w/o CRF) |    |
| Semi-sup. w/ image-level label | WideResNet38 | 75.1  (w/o CRF) | [link](https://drive.google.com/file/d/12bBCmvV8lPSmO3DGs4TpM9tplH4yHSAA/view?usp=sharing) |
| Semi-sup. w/o image-level label | WideResNet38 | 72.4 (w/o CRF) | [link](https://drive.google.com/file/d/11f43JVRWCWUb1vLlrtCI-yQP6Lvrii3s/view?usp=sharing) |

## Weakly Supervised Semantic Segmentation

#### Training

Train the weakly supervised model:
```bash
python tools/train_voc_wss.py --config experiments/slrnet_voc_wss_v100x2.yaml --run custom_experiment_run_id
```
For the COCO or L2ID dataset, please refer to the relevant scripts/configs with the suffix "_coco".

#### Inference and Evaluation
```bash
python tools/infer_voc.py \
       --config experiments/slrnet_voc_wss_v100x2.yaml \
       --checkpoint path/to/checkpoint20.pth.tar \
       --use_cls_label 0 \
       --output outputs/wss_prediction \
       --data_list voc12/val.txt \
       --data_root path/to/VOCdevkit/VOC2012 \
       --fp_cut 0.3 \
       --bg_pow 3
``` 
#### Two-stage Training 
Generate pseudo labels with following parameters:
```bash
...
       --data_list voc12/train_aug.txt \
       --use_cls_label 1 \
...
``` 
Then we train the DeepLabV3+ (w/ ResNet101 backbone) network implemented by [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

## Semi-Supervised Semantic Segmentation with Pixel-level and Image-level Labeled Data
#### Training
```bash
python tools/train_voc_semi.py --config experiments/slrnet_voc_semi_w_cls_v100x2.yaml --run custom_experiment_run_id
```
#### Inference and Evaluation
```bash
python tools/infer_voc.py \
       --config experiments/slrnet_voc_semi_w_cls_v100x2.yaml \
       --checkpoint path/to/checkpoint28.pth.tar \
       --use_cls_label 0 \
       --output outputs/semi_val_prediction \
       --data_list voc12/val.txt \
       --data_root ../VOCdevkit/VOC2012 \
       --fp_cut 0.3 \
       --bg_pow 1 \
       --apply_crf 0 \
       --verbose 0
``` 
## Semi-Supervised Semantic Segmentation with Pixel-level Labeled and Unlabeled Data
#### Training
```bash
python tools/train_voc_semi.py --config experiments/slrnet_voc_semi_wo_cls.yaml --run custom_experiment_run_id
```
#### Inference and Evaluation
```bash
python tools/infer_voc.py \
       --config experiments/slrnet_voc_semi_wo_cls.yaml \
       --checkpoint path/to/checkpoint32.pth.tar \
       --use_cls_label 0 \
       --output outputs/semi_val_prediction \
       --data_list voc12/val.txt \
       --data_root ../VOCdevkit/VOC2012 \
       --fp_cut 0.3 \
       --bg_pow 1 \
       --apply_crf 0 \
       --verbose 0
``` 

## Acknowledgements
We thank Nikita Araslanov, and Jiwoon Ahn for their great work that helped in the early stages of this project.
- [https://github.com/visinf/1-stage-wseg](https://github.com/visinf/1-stage-wseg)
- [https://github.com/jiwoon-ahn/irn](https://github.com/jiwoon-ahn/irn)

## Citation
We hope that you find this work useful. If you would like to acknowledge us, please, use the following citation:
```
@article{pan2022learning,
  title={Learning Self-supervised Low-Rank Network for Single-Stage Weakly and Semi-supervised Semantic Segmentation},
  author={Pan, Junwen and Zhu, Pengfei and Zhang, Kaihua and Cao, Bing and Wang, Yu and Zhang, Dingwen and Han, Junwei and Hu, Qinghua},
  journal={International Journal of Computer Vision},
  pages={1--15},
  year={2022},
  publisher={Springer}
}
```
## Contact
Junwen Pan <junwenpan@tju.edu.cn>