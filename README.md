
# ImmuneLLMs

ImmuneLLMs is a specialized framework integrating Large Language Models (LLMs) and Graph Neural Networks (GNNs) for biological and immunological research.

---

### Step-by-Step Environment Setup

```bash
# create environment
conda create -n ImmuneLLMs python=3.9 -y
conda activate ImmuneLLMs

# CUDA 11.8
pip install pyarrow==17.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install pandas==2.0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install numpy==1.24.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch_geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cu118.html -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install transformers==4.46.2 accelerate==1.0.1 peft==0.10.0 bitsandbytes==0.42.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install fair-esm==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install datasets==2.19.1 -i https://pypi.tuna.tsinghua.edu.cn/simple 
