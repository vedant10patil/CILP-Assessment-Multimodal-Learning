# CILP Assessment Results

This folder contains the experimental results and outputs generated from the four project notebooks. Below is a summary of the analysis performed in each file.

## Notebook Summaries

### 1. `01_dataset_exploration.ipynb`
* **Purpose:** Validated data integrity, visualized RGB-LiDAR pairs, and analyzed class distribution.
* **Key Finding:** Identified a severe class imbalance (**93% Cubes, 7% Spheres**), necessitating the use of weighted loss functions and data shuffling for subsequent tasks.

### 2. `02_fusion_comparison.ipynb`
* **Purpose:** Compared **Late Fusion** against three **Intermediate Fusion** strategies (Concatenation, Addition, Hadamard) on a 10% data subset.
* **Key Finding:** **Intermediate Fusion (Concatenation)** was the best performing architecture (**100% Accuracy**, 0.006 Loss), outperforming Late Fusion (94.88%) by effectively leveraging early spatial feature interactions.

### 3. `03_strided_conv_ablation.ipynb`
* **Purpose:** Analyzed the impact of downsampling layers by replacing fixed `MaxPool2d` with learnable `Strided Convolutions`.
* **Key Finding:** The baseline **MaxPool2d** proved superior for this specific sparse geometric task (**99.07% Accuracy**) compared to Strided Convolution (90.70%), likely due to better translation invariance and faster convergence on the limited subset.

### 4. `04_final_assessment.ipynb`
* **Purpose:** Trained the complete **Contrastive Image-LiDAR Pretraining (CILP)** pipeline.
* **Pipeline Stages:**
    1.  **Contrastive Pretraining:** Aligned RGB and LiDAR embeddings (Loss: **0.0444**).
    2.  **Projector Training:** Mapped RGB features to LiDAR space (MSE: **0.0012**).
    3.  **Final Classifier:** Classified shapes using projected features.
* **Key Finding:** Achieved **92.89% Accuracy** on the validation set, successfully demonstrating robust cross-modal feature transfer despite the heavy class imbalance.