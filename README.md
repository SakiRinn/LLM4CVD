# LLM4CVD: An Experimental Study

This is the codebase for the paper "Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study", which is submitted to SCIENCE CHINA Information Sciences (SCIS). The arxiv version of this work will be publicly available soon.

To facilitate related communities and encourage future studies, we provide an easy-to-use and unified codebase to implement three graph-based models, two medium-size BERT-based sequence models, and four LLMs to study their performance for the code vulnerability detection task. Our codebase is built on the top of some related codebases provided below (Awesome Helpful Resources).


## Investigated Model List

| Dataset | Venue | Type | Paper Link |
| --- | --- | --- | --- |
| Devign | NeurIPS  | Graph | [Link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html) |
| ReGVD | IEEE ICSE  | Graph | [Link](https://dl.acm.org/doi/abs/10.1145/3510454.3516865) |
| GraphCodeBERT | ICLR  | Graph | [Link](https://arxiv.org/abs/2009.08366) |
| CodeBERT | EMNLP  | Sequence | [Link](https://arxiv.org/abs/2002.08155) |
| UniXcoder | ACL  | Sequence | [Link](https://arxiv.org/abs/2203.03850) |
| Llama-2-7B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2307.09288) |
| CodeLlama-7B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2308.12950) |
| Llama-3-8B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2407.21783) |
| Llama-3.1-8B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2407.21783) |


## Dataset

We provide our converted datasets in our HuggingFace dataset repository.
At the current path, you can download the datasets by the following command, and remember to rename `VulResource` to `data`:

```shell
git clone https://huggingface.co/datasets/xuefen/VulResource
```

**Original paper and resources are listed below.**

| Dataset | Venue |  Paper Link |
| --- | --- | --- |
| ReVeal | IEEE TSE  | [Link](https://ieeexplore.ieee.org/abstract/document/9448435/?casa_token=S7Edzt0cuYkAAAAA:XId-rO6uAISCMYMyq4bvmcD83vqSfPCnZDqycv8iHI-tRZ9OVm-gAZzwIVZZGustUX1IsQ7Oew) |
| Devign | NeurIPS | [Link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html) |
| Draper | IEEE ICMLA |  [Link](https://arxiv.org/abs/1807.04320) |
| BigVul | IEEE ICSE | [Link](https://dl.acm.org/doi/abs/10.1145/3379597.3387501) |
| DiverseVul | IEEE ICSE  |  [Link](https://dl.acm.org/doi/abs/10.1145/3607199.3607242) |


## How to Run and Evaluate

We provided user-friendly shell scripts to simplify model training and evaluation. These scripts are located in the `scripts/` directory, and their general functionalities are as follows:

```shell
scripts
├── finetune.sh                     # Fine-tune LLMs for the experiments on Section 6.2 and 6.3.
├── inference.sh                    # Evaluate LLMs fine-tuned using `finetune.sh`.
├── finetune_imbalance.sh           # Fine-tune LLMs for the experiments on Section 6.4.
├── inference_imbalance.sh          # Evaluate LLMs fine-tuned using `finetune_imbalance.sh`.
├── finetune_ablation.sh            # Fine-tune LLMs for the experiments on Section 6.5.
├── inference_ablation.sh           # Evaluate LLMs fine-tuned using `finetune_ablation.sh`.
├── train.sh                        # Train graph-based and medium-size sequence models for the experiments on Section 6.2.
├── test.sh                         # Evaluate models trained using `train.sh`.
├── train_imbalance.sh              # Train graph-based and medium-size sequence models for the experiments on Section 6.4.
├── test_imbalance.sh               # Evaluate models trained using `train_imbalance.sh`.
└── to_graph.sh                     # Convert data into graph format as input for Devign model.
```

Before using these scripts, you need to:
1. Use the `cd` command to set the running directory to the root of this repository.
2. Place all data in the `data/` directory, following the directory structure and file names provided in our open-sourced HuggingFace repository.
3. Install all dependencies listed in the `requirements.txt` by running the command `pip install -r requirements.txt`.

The trained models and output log will be generated in the `outputs/` directory.

### Quick start

To quickly get started, you can run the following examples:

```shell
# For the experiments on Section 6.2.
./scripts/finetune.sh             reveal  llama3.1  0-512 16    0;
./scripts/inference.sh            reveal  llama3.1  0-512       0;
./scripts/train.sh                reveal  ReGVD     0-512       0;
./scripts/test.sh                 reveal  ReGVD     0-512       0;
# For the experiments on Section 6.3.
./scripts/finetune.sh             mix     llama3.1  128-256 32  0;
./scripts/inference.sh            mix     llama3.1  128-256     0;
./scripts/train.sh                mix     ReGVD     128-256     0;
./scripts/test.sh                 mix     ReGVD     128-256     0;
# For the experiments on Section 6.4.
./scripts/finetune_imbalance.sh   draper  llama3.1  0·2         0;
./scripts/inference_imbalance.sh  draper  llama3.1  0·2         0;
./scripts/train_imbalance.sh      draper  CodeBERT  0·2         0;
./scripts/test_imbalance.sh       draper  CodeBERT  0·2         0;
# For the experiments on Section 6.5.
./scripts/finetune_ablation.sh    reveal  llama3.1  8 16        0;
./scripts/inference_ablation.sh   reveal  llama3.1  8 16        0;
```

You can modify the command-line arguments in the above examples to perform other experiments mentioned in the paper.

Specifically, the second parameter represents the dataset name, which corresponds to the folder name in the `data/` directory.
You can customize a new dataset (assume it is named `xxx`) by following the template of our open-sourced dataset on the HuggingFace repository. Store it according to the following file structure:

```
data
└── xxx
    └── alpaca
        ├── xxx_0-123_test.json
        ├── xxx_0-123_train.json
        ├── xxx_0-123_validate.json
        ├── xxx_123-456_test.json
        ├── xxx_123-456_train.json
        ├── xxx_123-456_validate.json
        └── ...
```

The third parameter specifies the model name, which has the presetting supported values:

- For scripts prefixed with `finetune` and `inference`, the supported values are: `llama-2`, `codellama`, `llama-3`, and `llama-3.1` (all lowercase).
- For scripts prefixed with `train` and `test`, the supported values are: `Devign`, `ReGVD`, `GraphCodeBERT`, `CodeBERT`, and `UniXcoder` (case-sensitive).

Other parameters can refer to the usage within every script.

### Run Devign model

Unlike other models, the Devign model requires data to be converted into graph format before training and evaluation. To simplify this process, we provide the `to_graph.sh` script.

Before converting, you need to download `joern.zip` from [this link](https://), extract it, and store all the files in the `joern/` directory. Make sure the current user has execute (`x`) permissions for `joern-parse`.

An example of training and evaluating the Devign model is as follows:

```shell
./scripts/to_graph.sh   reveal  0-512;
./scripts/train.sh      reveal  Devign  0-512   0;
```


## Awesome Helpful Resources

We implement our studied models by referencing the following resources or codebases, and we also recommend some useful related resources for further study.

| Resource Name | Summary | Link |
| --- | --- | --- |
| VulLLM | Referenced Codebase for Implementation | [Link](https://ieeexplore.ieee.org/abstract/document/9448435/?casa_token=S7Edzt0cuYkAAAAA:XId-rO6uAISCMYMyq4bvmcD83vqSfPCnZDqycv8iHI-tRZ9OVm-gAZzwIVZZGustUX1IsQ7Oew) |
| Devign | Referenced Codebase for Implementation | [Link](https://github.com/saikat107/Devign) |
| CodeBERT Family | Referenced Codebase for Implementation | [Link](https://github.com/microsoft/CodeBERT) |
| ReGVD | Referenced Codebase for Implementation | [Link](https://github.com/daiquocnguyen/GNN-ReGVD) |
| Llama Family | Meta AI Open-source LLMs | [Link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html) |
| Evaluate ChatGPT for CVD | Recommened Codebase | [Link](https://github.com/soarsmu/ChatGPT-VulDetection) |
| Awesome Code LLM | Recommened Paper List | [Link](https://github.com/PurCL/CodeLLMPaper) |
| Awesome LLM for Software Engineering | Recommened Paper List | [Link](https://github.com/gai4se/LLM4SE) |
| Awesome LLM for Security | Recommened Paper List | [Link](https://github.com/liu673/Awesome-LLM4Security) |


## Acknowledgement

We are very grateful that the authors of VulLLM, CodeLlama, Meta AI and other open-source efforts which make their codes or models publicly available so that we can carry out this experimental study on top of their hard works.


## Citing this work
If you find this codebase useful in your research, please consider citing our work and previous great works as follows.
By the way, collaboration and pull requests are always welcome! If you have any questions or suggestions, please feel free to contact us : )

```bibtex
@article{jiang2024investigating,
  title={Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study},
  author={Jiang, Xuefeng and Wu, Lvhua and Sun, Sheng and Li, Jia and Xue, Jingjing and Wang, Yuwei and Wu, Tingting and Liu, Min},
  journal={arXiv preprint},
  year={2024}
}

@article{feng2020codebert,
  title={Codebert: A pre-trained model for programming and natural languages},
  author={Feng, Zhangyin and Guo, Daya and Tang, Duyu and Duan, Nan and Feng, Xiaocheng and Gong, Ming and Shou, Linjun and Qin, Bing and Liu, Ting and Jiang, Daxin and others},
  journal={arXiv preprint arXiv:2002.08155},
  year={2020}
}

@article{du2024generalization,
  title={Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning},
  author={Du, Xiaohu and Wen, Ming and Zhu, Jiahao and Xie, Zifan and Ji, Bin and Liu, Huijun and Shi, Xuanhua and Jin, Hai},
  journal={arXiv preprint arXiv:2406.03718},
  year={2024}
}
```
