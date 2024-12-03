# LLM4CVD: An Experimental Study

This is the codebase for the paper "Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study", which is submitted to SCIENCE CHINA Information Sciences (SCIS). The arxiv version of this work will be publicly available soon.

To facilitate related communities and encourage future studies, we provide an easy-to-use and unified codebase to implement three graph-based models, two medium-size BERT-based sequence models, and four LLMs to study their performance for the code vulnerability detection task. Our codebase is built on the top of some related codebases provided below (Awesome Helpful Resources).

## Investigated Model List

| Dataset | Venue | Type  |   Paper Link |
| --- | --- | --- | --- |
| Devign | NeurIPS  | Graph | [Link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html) |
| ReGVD | IEEE ICSE  | Graph | [Link](https://dl.acm.org/doi/abs/10.1145/3510454.3516865) |
| GraphCodeBERT | ICLR  | Graph | [Link](https://arxiv.org/abs/2009.08366) |
| CodeBERT | EMNLP  | Sequence | [Link](https://arxiv.org/abs/2002.08155) |
| UniXcoder | ACL  | Sequence | [Link](https://arxiv.org/abs/2203.03850) |
| Llama-2-7B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2307.09288) |
| CodeLlama-7B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2308.12950) |
| Llam-3-8B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2407.21783) |
| Llama-3.1-8B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2407.21783) |


## Dataset

We provide our converted datasets in our HuggingFace dataset repository.
You can download the datasets by the following command:

```shell
huggingface-cli download -d xuefen/VulResource
```

**Original paper and resources are listed below.**

| Dataset | Venue |  Paper Link |
| --- | --- | --- |
| ReVeal | IEEE TSE  | [Link](https://ieeexplore.ieee.org/abstract/document/9448435/?casa_token=S7Edzt0cuYkAAAAA:XId-rO6uAISCMYMyq4bvmcD83vqSfPCnZDqycv8iHI-tRZ9OVm-gAZzwIVZZGustUX1IsQ7Oew) |
| Devign | NeurIPS | [Link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html) |
| Draper | IEEE ICMLA |  [Link](https://arxiv.org/abs/1807.04320) |
| BigVul | IEEE ICSE | [Link](https://dl.acm.org/doi/abs/10.1145/3379597.3387501) |
| DiverseVul | IEEE ICSE  |  [Link](https://dl.acm.org/doi/abs/10.1145/3607199.3607242) |

## Requirements
TBD



## How to Run and Evaluate
TBD

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
