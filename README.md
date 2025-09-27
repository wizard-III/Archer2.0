![status](https://img.shields.io/badge/status-WIP-yellow)  

# 🚧 Work in Progress

This project is currently under development. Features, structure, and documentation may change frequently.


<div align="center">

# ✨ Archer2.0

<div>
🏹️  Reinforcement Learning for Enhanced Reasoning in LLMs  🎯
</div>

</div>
<div>
<br>

<div align="center">

[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://rogue-canopy-54a.notion.site/Asymmetric-Dual-Clipping-Policy-Optimization-Escaping-Local-Optima-Unlocking-the-Full-Potential--2650e4c8c16a8034a5d3dfec358c9021)
[![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/wizard-III/Archer2.0)
[![Model](https://img.shields.io/badge/Model-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/Fate-Zero/Archer2.0-Code-1.5B-Preview)
[![Data](https://img.shields.io/badge/Data-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/datasets/Fate-Zero/Archer2.0-Code-1.5B)
[![知乎](https://img.shields.io/badge/知乎-0084FF?style=for-the-badge&logo=zhihu&logoColor=white)](https://zhuanlan.zhihu.com/p/1950989602023244983)

</div>


## Getting Started

### Installation

```bash
# Installing Python 3.10 Environment.
conda create -n archer python=3.10 -y
conda activate archer

# Installing dependencies.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --no-cache-dir flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

cd ArcherCodeR
pip install -e .
```

### Data Preparation

Download the training and test data from Hugging Face.

```bash
python tools/download_datasets.py
```

#### Initialize Ray Cluster

We have provided a one-click script to initialize Ray environments on any number of machines. Run the following command on the head node:

```bash
bash ./tools/start_ray.sh
```

Note: 
- Please replace your_wandb_api_key in export WANDB_API_KEY=your_wandb_api_key with your actual key.
- Hostfile locations vary across operating systems (e.g., on my machine, it's located at /etc/mpi/hostfile). Locate the file on your server and modify its content accordingly.

### Training

We have currently only provided the script and data to reproduce the results of the “Archer-Code-1.5B”.

```bash
bash ./scripts/train/run_archer_qwen2.5_1.5b_code.sh
```

### Evaluation

#### Step 1: Convert model format

Run the following command to convert the model to Hugging Face format:

```bash
bash ./tools/model_merge.sh
```

#### Step 2: Run evaluation

Execute the script below to evaluate model performance on the LiveCodeBench v5 benchmark:

```bash
bash ./scripts/eval/run_eval.sh
```

Note: Please update the path parameters in the scripts above as needed.

## Technical Report
[Asymmetric Dual-Clipping  —— Escaping Local Optima, Unlocking the Full Potential of RL for LLMs](https://rogue-canopy-54a.notion.site/Asymmetric-Dual-Clipping-Escaping-Local-Optima-Unlocking-the-Full-Potential-of-RL-for-LLMs-2650e4c8c16a8034a5d3dfec358c9021)
[Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR](https://arxiv.org/abs/2507.15778)

## Acknowledgements

- We build our model upon [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
- Training was carried out with a modified version of [verl](https://github.com/volcengine/verl).

## Citation

Please cite the following:
```bibtex
@misc{ACPO2025,
  title={Asymmetric Clipping Policy Optimization},
  author={Wang, Jiakang and Liu, Runze and Lin, Lei and Hu, Wenping and Li, Xiu and Zhang, Fuzheng and Zhou, Guorui},
  howpublished={\url{https://rogue-canopy-54a.notion.site/Asymmetric-Dual-Clipping-Unleashing-the-Full-Potential-of-RL-in-LLM-Training-2650e4c8c16a8034a5d3dfec358c9021}},
  note={Notion Blog},
  year={2025}
}
```

```bibtex
@article{wang2025stabilizing,
  title={Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR},
  author={Wang, Jiakang and Liu, Runze and Zhang, Fuzheng and Li, Xiu and Zhou, Guorui},
  journal={arXiv preprint arXiv:2507.15778},
  year={2025}
}
```

