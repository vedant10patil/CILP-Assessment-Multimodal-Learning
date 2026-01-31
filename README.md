# General
**Project title**: CILP Assessment: Multimodal Learning

**Description**: This project contains the submission for Lab 2 of the course *Applied Hands-On Computer Vision*.
It uses material from the *NVIDIA DLI Multimodality* course. The goal is to train a classifier on a dataset containing RGB and LiDAR data of spheres and cubes.
We test multiple fusion strategies and downsampling methods.

Project structure:
```
├── notebooks/
│   ├── 01_dataset_exploration.ipynb    # Task 2 - Dataset creation
│   ├── 02_fusion_comparison.ipynb      # Task 3 - Comparision of fusion strategies
│   ├── 03_strided_conv_ablation.ipynb  # Task 4 - Comparision of downsampling methods
│   └── 04_final_assessment.ipynb       # Task 5 - Final classifier training
│
├── src/
│   ├── __init__.py
│   ├── models.py          # All model architectures
│   ├── datasets.py        # Dataset classes
│   ├── training.py        # Training loops
│   ├── visualization.py   # Plotting utilities
│   └── utils.py           # Functions for seeding, inference and helper
│
├── checkpoints/           # Saved model weights
├── results/               # Figures and tables
├── requirements.txt       # Dependencies
└── README.md              # Setup and usage instructions
```


# How to Run the Code
This project contains four notebooks.
The first notebook creates the dataset as a FiftyOne dataset.
The data should be stored in the following structure:
```
data/assessment/
├── cubes/
│   ├── rgb/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   └── lidar/
│       ├── 0000.npy
│       ├── 0001.npy
│       └── ...
└── spheres/
    ├── rgb/
    └── lidar/
```

**Note:**  For spheres, the lidar folder is duplicated.
This is due to a copy error that was not corrected because of high latency when working with Google Drive.
The dataset creation notebook randomly selects 10% of samples for each modality independently. It also creates a train/test split.
The dataset is saved locally.
All remaining notebooks can run independently after the first one.
At the top of each notebook we mount Google Drive, install dependencies, and copy files to the working directory for faster access.
The dependencies are listed in requirements.txt.

The code is seeded before each training loop and during DataLoader creation.
This makes the results reproducible. However, small differences may still occur due to different CUDA versions on Colab.
Each notebook includes a summary of the machine used.


# Weights & Biases
We used Weights & Biases (wandb) to monitor training: https://wandb.ai/vedantanilpatil-university-of-potsdam/cilp-extended-assessment


# Summary of results
### 01_dataset_exploration.ipynb
* **Total Samples:** 10,751 image-LiDAR pairs.
* **Class Balance:** Highly imbalanced — **Cubes (93%)**, **Spheres (7%)**.
* **Data Split:** Used a 10% subset (approx. 1,075 samples) for rapid experimentation.

### 02_fusion_comparison.ipynb
Comparing fusion strategies on a 10% subset (15 epochs).
* **Intermediate Fusion (Concat):** **100.00% Accuracy** (Best Performer).
* **Intermediate Fusion (Hadamard):** 100.00% Accuracy.
* **Intermediate Fusion (Add):** 98.14% Accuracy.
* **Late Fusion:** 94.88% Accuracy.

### 03_strided_conv_ablation.ipynb
Comparing downsampling methods on the Intermediate Fusion (Concat) model.
* **MaxPool2d (Baseline):** **99.07% Accuracy** (Converged faster).
* **Strided Convolution:** 90.70% Accuracy.

### 04_final_assessment.ipynb
Final CILP pipeline performance (Pretrain → Project → Classify).
* **Contrastive Pretraining Loss:** **0.0444** (Target < 3.5 )
* **Projector MSE Loss:** **0.0012** (Target < 2.5 )
* **Final Classifier Accuracy:** **93.40%** (High performance despite 13:1 class imbalance).
## Classifier Results
- Contrastive Pretraining Validation loss: 0.0444
- Projector Validation MSE: 0.0012
- RGB to Lidar Classifier Validation accuracy: 92.89%

## Reproducibility
Data Handling: Experiments utilize automatic localized data extraction from assessment.zip to ensure high-speed training on Google Colab.

## References
NVIDIA DLI Multimodality Workshop

Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)