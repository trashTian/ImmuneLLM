# Baselines Implementation & Benchmarking

This directory contains the re-implementation and evaluation scripts for the main baselines compared in the **ImmuneLLM** paper. For transparency and reproducibility, we provide our execution code along with links to the original official repositories.

## 📂 Included Baselines

| Model | Category | Source |
| :--- | :--- | :--- |
| **T-SCAPE** | Unified Modeling |  https://github.com/seoklab/T-SCAPE |
| **TEIM** | TCR-Peptide |  https://github.com/pengxingang/TEIM |
| **DeepAntigen** | Unified Modeling |  https://github.com/JiangBioLab/deepAntigen|
| **UnifyImmun** | Unified Modeling |  https://github.com/hliulab/unifyimmun |
| **TransPHLA** | HLA-Peptide |  https://github.com/Felipepuiggarimedici/TransPHLA|
| **ESM2-3B (FT)** | PLM-based |  https://github.com/facebookresearch/esm|
| **ESM2-3B (Frozen)** | PLM-based | https://github.com/facebookresearch/esm|


## ⚖️ Evaluation Protocol: The Unified Benchmark

To ensure a rigorous and fair comparison, we adopted the following protocol:
- **Unified Training & Testing**: All baselines (unless specified below) were re-trained and evaluated on the same benchmark datasets as used in **UnifyImmun**.
- **Statistical Reliability**: Every result reported in the main paper is derived from **5-fold bootstrap sampling**. We report the mean performance along with the standard deviation to ensure statistical significance.

## 🛠️ Model-Specific Implementations & Constraints

Due to the inherent design limitations of certain baseline models, we applied specific data processing or evaluation strategies:

### 1. T-SCAPE
- **Training**: Since the official training source code is not publicly available, we conducted our evaluation using the **official best-performing checkpoints**.
- **Length Constraints**: As per the original paper, T-SCAPE supports a maximum peptide length of 20. 
    - For peptides shorter than 20 residues, zero-padding was applied. 
    - Peptides exceeding 20 residues were excluded from the T-SCAPE specific test subset, as the architecture cannot process them.

### 2. TEIM
- **Data Filtering**: The TEIM architecture requires specific sequence length ranges: **CDR3 (7–20 residues)** and **Epitopes (8–11 residues)**.
- **Consistency**: To maintain a fair "head-to-head" comparison, we applied these length constraints across all datasets during the TEIM benchmarking phase, ensuring that the test set remains consistent across all models being compared in those specific scenarios.

