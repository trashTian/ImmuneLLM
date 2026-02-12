# Data & Knowledge Repository

This directory contains the datasets and generation scripts for **ImmuneLLM**.

## 🔗 Data Download (Pre-processed)

The final QA pairs for training and evaluation are available via the link below. These include the basic binding pairs and the reasoning-enhanced datasets.
- **Download Link**: [ImmuneLLM Processed Data](https://pan.baidu.com/s/1f6OSt_Dh2WCrUtqx62mkpQ?pwd=8sws)
- **Extraction Code**: `8sws`

## 📂 Directory Structure

| Directory | Description |
| :--- | :--- |
| `unifyimmun_benchmark_data/` | Original source data from the UnifyImmun benchmark. |
| `original_data_preprocess/` | Scripts for cleaning and converting raw data into basic QA format. |
| `knowledge_generate/` | Code for generating Phase A (Symbolic) and Phase B (Reasoning) QA pairs. |
| `rules/` | The induced biophysical ruleset (Phase A) used for mechanistic alignment. |



## 🛠 Usage Overview

1. **Baseline Data**: Access the raw UnifyImmun benchmark in `unifyimmun_benchmark_data/`.
2. **Preprocessing**: Use `original_data_preprocess/` to build the foundational binding-prediction QA pairs.
3. **Knowledge Expansion**: Use `knowledge_generate/` to reproduce the symbolic and reasoning-based data used in Phase A & B.
4. **Rules**: Review the induced biophysical constraints in `rules/`, which guide the model's mechanistic inference.
