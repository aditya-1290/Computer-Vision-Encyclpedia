# Computer Vision Encyclopedia

## Overview
This repository is a comprehensive encyclopedia of computer vision techniques, from foundational classical image processing to advanced deep learning architectures and applications. It is designed to provide both theoretical understanding and practical implementations.

## Visual Table of Contents
- I. Image Fundamentals
- II. Classical Image Processing
- III. Feature Detection & Description
- IV. Image Segmentation (Classical)
- V. Deep Learning Basics for CV
- VI. Convolutional Neural Networks Architectures
- VII. CV Applications & Tasks
- VIII. Other Key Techniques
- IX. Projects and Mini Apps

## Prerequisites
- Python 3.8+
- Libraries: numpy, opencv-python, matplotlib, scikit-image, scikit-learn, torch, torchvision, tensorflow, Pillow, albumentations

## How to Use
- Explore each module folder for theory, math, and code implementations.
- Classical algorithms are implemented from scratch using NumPy/OpenCV.
- Deep learning models use PyTorch and TensorFlow.
- Projects combine multiple techniques for practical applications.

## Installation
Install required packages using:

```bash
pip install -r requirements.txt
```

## Contribution
Contributions are welcome! Please follow the structure and philosophy outlined in the repository.

## Project Structure

### Folders

- **i_image_fundamentals/**: Basic concepts and fundamentals of image processing, including pixel manipulation, color spaces, and image properties.
- **ii_classical_image_processing/**: Implementations of classical image processing techniques such as filtering, morphological operations, edge detection, and image enhancement.
- **iii_feature_detection_description/**: Feature detection and description algorithms including corner detection, blob detection, edge detection, and descriptors like SIFT, SURF, ORB.
- **iv_image_segmentation_classical/**: Classical image segmentation methods such as thresholding for separating objects from background.
  - **thresholding.py**: Implementation of thresholding techniques for image segmentation.
- **v_deep_learning_basics_cv/**: Fundamentals of deep learning applied to computer vision, including neural networks basics and training techniques.
- **vi_convolutional_neural_networks_architectures/**: Implementations of various convolutional neural network architectures like AlexNet, VGG, ResNet, Inception, EfficientNet, and Vision Transformer.
- **vii_cv_applications_tasks/**: Common computer vision applications and tasks including image classification, semantic segmentation, object detection (two-stage and one-stage), pose estimation, and generative models.
- **viii_other_key_techniques/**: Other important techniques such as model optimization (quantization, pruning, knowledge distillation), explainable AI (Grad-CAM, saliency maps), video processing (reading video, optical flow), and 3D computer vision (depth estimation, NeRFs, point clouds).
- **ix_projects_and_mini_apps/**: Projects and mini applications demonstrating computer vision concepts.
- **xi_projects_and_mini_apps/**: Additional projects and mini applications like image search engine, real-time object detection, photo restoration, style transfer, and OCR/document analysis.
- **study_materials/**: Supplementary study materials, notes, and resources for learning computer vision.
- **images/**: Image assets used in documentation, examples, and visualizations.
- **vision_env/**: Virtual environment configuration for the project dependencies.

### Files

#### Root Level Files
- **README.md**: This file providing an overview, installation instructions, and project structure.
- **requirements.txt**: List of Python dependencies required to run the project.
- **.gitignore**: Git ignore rules to exclude temporary files, environments, and large data from version control.
- **TODO.md**: Task list for project development and maintenance.

#### ii_classical_image_processing/
- **all_filters.png**: Visualization of various image filters applied to an image.
- **closing.png**: Result of morphological closing operation on an image.
- **convolution.py**: Python script implementing convolution operations for image filtering.
- **dilation.png**: Result of morphological dilation operation.
- **erosion.png**: Result of morphological erosion operation.
- **gaussian_blur.png**: Image after applying Gaussian blur filter.
- **gradients_(sobel).png**: Gradient magnitude using Sobel operator.
- **gradients_laplacian.png**: Laplacian gradient for edge detection.
- **gradients_laplacian.py**: Implementation of Laplacian gradient computation.
- **image_filtering.py**: Script for various image filtering techniques.
- **laplacian.png**: Output of Laplacian edge detection.
- **median_blur.png**: Image after median blur filtering.
- **morphological_operations.png**: Visualization of morphological operations.
- **morphological_ops.py**: Implementation of morphological operations like erosion, dilation, opening, closing.
- **opening.png**: Result of morphological opening operation.
- **original_binary.png**: Original binary image for morphological operations.
- **prewitt_edges.png**: Edges detected using Prewitt operator.
- **sharpened_image.png**: Image after sharpening filter.
- **sobel_edges.png**: Edges detected using Sobel operator.

#### iii_feature_detection_description/
- **bf_matches.png**: Feature matches using Brute Force matcher.
- **blob_detection.png**: Detected blobs in an image.
- **blob_detection.py**: Implementation of blob detection algorithms.
- **canny_comparison.png**: Comparison of Canny edge detection implementations.
- **canny_opencv.png**: Canny edges using OpenCV.
- **canny_scratch.png**: Canny edges implemented from scratch.
- **corner_detection.png**: Detected corners in an image.
- **corner_detection.py**: Implementation of corner detection algorithms like Harris and Shi-Tomasi.
- **dog_blobs.png**: Blobs detected using Difference of Gaussians (DoG).
- **edge_detection.py**: Various edge detection techniques.
- **feature_descriptors.png**: Visualization of feature descriptors.
- **feature_descriptors.py**: Implementation of feature descriptors like SIFT, SURF, ORB.
- **feature_matching.png**: Matched features between images.
- **feature_matching.py**: Implementation of feature matching algorithms.
- **flann_matches.png**: Feature matches using FLANN matcher.
- **harris_corners.png**: Corners detected using Harris corner detector.
- **homography_projection.png**: Image projection using homography.
- **log_blobs.png**: Blobs detected using Laplacian of Gaussian (LoG).
- **orb_descriptors.png**: ORB feature descriptors visualization.
- **shi_tomasi_corners.png**: Corners detected using Shi-Tomasi method.
- **sift_descriptors.png**: SIFT feature descriptors.
- **sift_keypoints.png**: SIFT keypoints detected in an image.
- **surf_descriptors.png**: SURF feature descriptors.

#### vi_convolutional_neural_networks_architectures/
- **alexnet.py**: Implementation of AlexNet CNN architecture.
- **vgg.py**: Implementation of VGG network architectures.
- **resnet.py**: Implementation of ResNet architectures with residual connections.
- **inception.py**: Implementation of Inception (GoogLeNet) architecture.
- **efficientnet.py**: Implementation of EfficientNet architectures.
- **vision_transformer.py**: Implementation of Vision Transformer (ViT) model.

#### vii_cv_applications_tasks/image_classification/
- **image_classification.py**: Implementation of image classification using deep learning models.

#### vii_cv_applications_tasks/semantic_segmentation/
- **fcn.py**: Implementation of Fully Convolutional Network for semantic segmentation.
- **unet.py**: Implementation of U-Net architecture for image segmentation.
- **deeplab.py**: Implementation of DeepLab models for semantic segmentation.

#### vii_cv_applications_tasks/object_detection/two_stage/
- **rcnn.py**: Implementation of Region-based Convolutional Neural Networks (R-CNN).
- **fast_rcnn.py**: Implementation of Fast R-CNN for object detection.
- **faster_rcnn.py**: Implementation of Faster R-CNN with region proposal network.
- **mask_rcnn.py**: Implementation of Mask R-CNN for instance segmentation.

#### vii_cv_applications_tasks/object_detection/one_stage/
- **yolo.py**: Implementation of YOLO (You Only Look Once) object detection.
- **ssd.py**: Implementation of Single Shot MultiBox Detector (SSD).
- **retinanet.py**: Implementation of RetinaNet with focal loss.

#### vii_cv_applications_tasks/pose_estimation/
- **openpose.py**: Implementation of OpenPose for human pose estimation.
- **movenet.py**: Implementation of MoveNet for pose estimation.

#### vii_cv_applications_tasks/generative_models/
- **autoencoders.py**: Implementation of autoencoders for generative modeling.
- **gans.py**: Implementation of Generative Adversarial Networks (GANs).
- **diffusion_models.py**: Implementation of diffusion models for image generation.

#### viii_other_key_techniques/model_optimization/
- **quantization.py**: Techniques for model quantization to reduce size and improve inference speed.
- **pruning.py**: Implementation of model pruning for efficiency.
- **knowledge_distillation.py**: Knowledge distillation methods for model compression.

#### viii_other_key_techniques/explainable_ai_xai/
- **grad_cam.py**: Implementation of Grad-CAM for visualizing CNN decisions.
- **saliency_maps.py**: Generation of saliency maps for model interpretability.

#### viii_other_key_techniques/video_processing/
- **reading_video.py**: Code for reading and processing video files.
- **optical_flow.py**: Implementation of optical flow estimation.

#### viii_other_key_techniques/3d_computer_vision/
- **depth_estimation.py**: Methods for estimating depth from images.
- **nerfs.py**: Implementation of Neural Radiance Fields (NeRFs) for 3D reconstruction.
- **point_clouds.py**: Processing and visualization of 3D point clouds.

#### xi_projects_and_mini_apps/
- **image_search_engine.py**: Mini application for image search using computer vision.
- **real_time_object_detection.py**: Real-time object detection application.
- **photo_restoration.py**: Photo restoration using deep learning techniques.
- **style_transfer_app.py**: Neural style transfer application.
- **ocr_and_document_analysis.py**: Optical Character Recognition and document analysis tools.
