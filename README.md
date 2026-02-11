# ImmuneLLM: A Unified Multimodal LLM Framework for T-cell Antigen Binding Specificity to HLA and TCR Molecules

Accurate in silico prediction of T-cell antigen binding specificity is a key challenge in computational biology, critical for vaccine design and cancer immunotherapy. Existing discriminative methods achieve strong performance by learning sequence patterns but often overlook physicochemical semantics and domain knowledge underlying molecular interactions. Furthermore, effectively harnessing generative reasoning for interpretable, mechanism-aware prediction remains under-explored in T-cell antigen binding specificity tasks. To address these gaps, we propose \textbf{ImmuneLLM}, a neuro-symbolic generative framework that unifies peptide--HLA and peptide--TCR binding. By coupling a pre-trained protein encoder with a Large Language Model, we develop a dual-phase knowledge-enhanced instruction tuning strategy. This approach first employs Symbolic Knowledge Induction to derive explicit biophysical rules, followed by Iterative Rationale Refinement to align reasoning chains with these logical constraints. Such a design enables the model to not only predict binding specificity but also elucidate underlying biochemical mechanisms via mechanistic inference. Extensive evaluations show that ImmuneLLM achieves SOTA performance across multiple benchmarks. Notably, experimental results demonstrate superior data efficiency; even when evaluated on out-of-distribution benchmarks, ImmuneLLM performs on par with discriminative baselines while requiring only a fraction of the training data.
<img width="3228" height="1107" alt="image" src="https://github.com/user-attachments/assets/36b3c001-8741-495e-a30d-bb10077323bd" />

---

## 🌟 Key Features
- **Phase A: Symbolic Knowledge Induction**: Explicit biophysical rule integration.
- **Phase B: Iterative Rationale Refinement**: Logic-aligned reasoning chain generation.
- **Mechanistic Interpretability**: Audit trails for biophysical decision-making.


---
### Step-by-Step Environment Setup

```bash
# create environment
conda create -n immunellm python=3.10 -y
conda activate immunellm

# CUDA 11.8
pip install pyarrow==17.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install pandas==2.0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install numpy==1.24.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install transformers==4.57.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install fair-esm==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install datasets==2.19.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install accelerate==1.12.0 peft==0.18.0  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
