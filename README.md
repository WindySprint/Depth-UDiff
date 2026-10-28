# Depth-UDiff
This is the project of paper "Depth-UDiff: A Depth-Prior Diffusion Model for Underwater Image Enhancement".

## Abstract
Existing underwater image enhancement methods employ handcrafted prior knowledge and general large vision models to mitigate regional quality variations, often resulting in structural inconsistencies and color distortion in spatially non-uniform degradation scenarios. Therefore, scene-adaptive priors and region-specific enhancement are two major challenges. To address this, we propose a depth-prior diffusion enhancement model (Depth-UDiff), which aims to guide underwater diffusion reconstruction through a multi-scene prior generation mechanism. Depth-UDiff comprises three key stages: (1) SAM-guided depth estimation utilizes degradation representations and segmentation cues to capture the environmental quality distribution, thereby estimating scene-related depth maps. (2) Depth-aware degradation correction decouples degradation features into structural and color components to perform depth-guided adaptive compensation, effectively mitigating the coupling interference between detail restoration and color correction. (3) Priors-based diffusion enhancement incorporates depth and correction priors to constrain the generation process, enabling natural visual reconstruction in complex degradation conditions. Extensive experiments on six benchmarks demonstrate that our method outperforms state-of-the-art methods in both quantitative metrics and visual quality. Code is available at: https://github.com/WindySprint/Depth-UDiff.

## Environment
```
1. Python 3.8.18
2. PyTorch 1.13.1
3. Torchvision 0.14.1
4. OpenCV-Python 4.8.1.78
5. NumPy 1.24.4
```

## Experimental Data
The experimental data is publicly available on [Google Drive](https://drive.google.com/drive/folders/1gmhaC5IMwk8vZ_NNH9Jw0ghP0azRKtt3).

## Checkpoints
The pretrained models are publicly available on [Google Drive](https://drive.google.com/drive/folders/1KzFr4nOsnLpNHUG29HOegBSbFZ9WCwNV?usp=sharing).

We use the pretrained SAM ViT-H model for depth estimation:
- **Source code:** [Segment Anything](https://github.com/facebookresearch/segment-anything)
- **Checkpoint:** [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
 
Download the SAM ViT-H checkpoint and place it in `./sam/weights/`, or update `sam_checkpoint` to its local path.

## Test
```
Generate enhanced images
1. Clone the repository
2. Place the pretrained depth and enhancement models in 'checkpoint_path'/'dnet_name'/model_best.pth and 'checkpoint_path'/'net_name'/model_best.pth
3. Download the SAM ViT-H checkpoint and set 'sam_checkpoint' in test.py
4. Put the test images in folder A/dataset_name
5. Set 'raw_images_path' to A and update 'dataset_name', 'dnet_name', and 'net_name' in test.py
6. Set 'result_path' and ensure the folder exists
7. Run python test.py
8. Find enhanced images in 'result_path'/'net_name'/'dataset_name'
```

```
Generate depth maps only
1. Place the pretrained depth model in 'checkpoint_path'/'dnet_name'/model_best.pth
2. Set 'sam_checkpoint' in test_depth.py
3. Put the images in folder A/dataset_name
4. Set 'raw_images_path' to A and update 'dataset_name' and 'dnet_name' in test_depth.py
5. Set 'result_path' and ensure the folder exists
6. Run python test_depth.py
7. Find depth maps in 'result_path'/'dnet_name'/'dataset_name'
```

## Train
```
Train the depth estimation network
1. Put the original images and corresponding reference depth maps in folders A and B
2. Set 'raw_images_path' to A and 'gt_images_path' to B in train_depth.py
3. Download the SAM ViT-H checkpoint and set 'sam_checkpoint'
4. Set 'dnet_name' and 'checkpoint_path', and ensure 'checkpoint_path' exists
5. Run python train_depth.py
6. Find the trained depth model in 'checkpoint_path'/'dnet_name'
```

```
Train the image enhancement network
1. Use the trained depth model and test_depth.py to generate depth maps for the enhancement training and validation images
2. Put the original images, corresponding GT images, and generated depth maps in folders A, B, and C
3. Set 'raw_images_path' to A, 'gt_images_path' to B, and 'd_images_path' to C in train.py
4. Set 'net_name' and 'checkpoint_path', and ensure 'checkpoint_path' exists
5. Run python train.py
6. Find the trained enhancement model in 'checkpoint_path'/'net_name'
```

# Contact

If you have any questions, please contact:

**Zhixiong Huang**  
Dalian University of Technology  
📧 hzxcyanwind@mail.dlut.edu.cn

