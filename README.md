# 4SP: 4-Stage Pipeline for Monocular Spacecraft Pose Estimation

[![Conference](https://img.shields.io/badge/IAC-2025-blue)](https://www.iafastro.org/events/iac/iac-2025/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)

## Overview

This repository contains the official implementation of the research paper presented at the **76th International Astronautical Congress (IAC 2025)** by **Daniel Pagliaricci**.

**4SP (4-Stage Pipeline)** is a novel deep learning framework designed for **3D Pose Estimation of Non-Cooperative Targets** (such as satellites and space debris) using a single monocular camera. The system addresses the "sim-to-real" gap and the challenge of high-dynamic lighting conditions in space by breaking down the pose estimation problem into four specialized, sequential stages.

<p align="center">
  <!-- Placeholder for a main architecture diagram or result image -->
  <img src="docs/assets/pipeline_overview.png" alt="4SP Pipeline Overview" width="800">
  <br>
  <em>Figure 1: High-level overview of the 4-Stage Pipeline architecture.</em>
</p>

## Architecture

The methodology relies on a modular divide-and-conquer approach, where each stage specializes in a specific abstraction level of the visual data:

### 1. Contour Segmentation
*   **Goal:** precise semantic segmentation of the spacecraft body to refine pose alignment.
*   **Model:** **Mask2Former** (Swin-Base backbone).
*   **Output:** Binary segmentation mask (Object vs. Background).
*   **Key Tech:** Fine-tuned Transformer-based segmentation for robust edge detection against Earth/Space backgrounds.

### 2. Visible Keypoints Detection
*   **Goal:** Detect semantic keypoints visible in the 2D image.
*   **Model:** Modified **ResNet50 (U-Net style)** with upsampling layers.
*   **Output:** Heatmaps and 2D coordinates for visible features.
*   **Key Tech:** Focal BCE Loss + SoftArgMax for differentiable coordinate extraction.

### 3. Invisible Keypoints Estimation
*   **Goal:** Infer the position of occluded or internal keypoints based on the visible ones.
*   **Model:** Fully Connected Network (DNN) with **Residual Connections**.
*   **Output:** Complete set of 2D keypoints (visible + occluded).
*   **Key Tech:** Weighted Masked Euclidean Loss to handle geometric uncertainty.

### 4. Rotation & Pose Regression
*   **Goal:** Estimate the final 3D orientation (Quaternion).
*   **Model:** Hybrid DNN regressor processing geometric features (vertices, edges, faces).
*   **Output:** Unit Quaternion (4D).
*   **Key Tech:** **Geodesic Loss** for accurate rotational error minimization.


## Repository Structure

```bash
4SP/
├── Contour/                         # Stage 1: Segmentation (Mask2Former)
│   ├── train_contour.py
│   └── eval_contour.py
├── Visible_Keypoints/               # Stage 2: Heatmap regression
│   ├── train_vis.py
│   ├── preprocess_vis.py
│   └── use_vis.py
├── Invisible_Keypoints/             # Stage 3: Geometric inference
│   ├── train_invis.py
│   ├── preprocess_invis.py
│   └── use_invis.py
├── Rotation/                        # Stage 4: Quaternion regression
│   ├── train_rotation.py
│   └── use_rotation.py
└── Database/                        # Synthetic data generation tools
```

## Installation

```bash
git clone https://github.com/yourusername/4SP.git
cd 4SP

# It is recommended to use a virtual environment (Conda or venv)
pip install -r Contour/requirements.txt
# Additional requirements for TF stages
pip install tensorflow matplotlib scipy numpy
```

## Usage

### Data Preparation
The pipeline expects synthetic data generated using the scripts in `Database/`. The data should be preprocessed before training each stage.

### Training

To train the pipeline from scratch, follow the sequential order:
**1. Train Contour Segmentor:**
```bash
cd Contour
python preprocess_contour.py
python train_contour.py   # Fine-tunes Mask2Former
```

**2. Train Visible Keypoints:**
```bash
cd Visible_Keypoints
python preprocess_vis.py  # Prepare NPZ files
python train_vis.py       # Train ResNet50 U-Net
```

**3. Train Invisible Keypoints:**
```bash
cd Invisible_Keypoints
python preprocess_invis.py
python train_invis.py
```

**4. Train Rotation Regressor:**
```bash
cd Rotation
python preprocess_rotation.py
python train_rotation.py
```

### Inference
You can use the `use_*.py` scripts in each directory to run inference on new images.

## 📊 Results

The method achieves state-of-the-art performance on synthetic benchmarks, significantly reducing angular error compared to direct regression methods.

| Metric | Stage | Value |
| :--- | :--- | :--- |
| **Masked MAE** | Visible Keypoints | *See paper* |
| **IoU** | Contour Segmentation | *See paper* |
| **Geodesic Error** | Final Rotation | *< 0.30° (example)* |

<p align="center">
  <img src="docs/assets/results_sample.png" alt="Qualitative Results" width="800">
  <br>
  <em>Figure 2: Qualitative results showing keypoint detection and final wireframe alignment.</em>
</p>

## Citation

If you use this code or research in your work, please cite the IAC 2025 paper:

```bibtex
@inproceedings{pagliaricci20254sp,
  title={A Multi-Stage Deep Neural Network Approach for Angular Pose Estimation in Nanosatellite Capture.},
  author={Pagliaricci, Daniel},
  booktitle={76th International Astronautical Congress (IAC)},
  year={2025},
  location={Sydney, Australia}
}
```

## Acknowledgements

This work was supported by:
- National Council for Scientific and Technological Development (CNPq)
- National Fund for Scientific and Technological Development (FNDCT)
- Ministry of Science, Technology and Innovations (MCTI)

Process No. 407721/2022-3

---

## License

This project is part of ongoing research at the Federal University of Sao Carlos (UFSCar), Department of Electrical Engineering.
