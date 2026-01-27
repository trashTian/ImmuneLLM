
# ImmuneLLMs

ImmuneLLMs is a specialized framework integrating Large Language Models (LLMs) and Protein Language Models (PLMs) for biological and immunological research.

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

pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install fair-esm==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install datasets==2.19.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install accelerate==1.12.0 peft==0.18.0  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
