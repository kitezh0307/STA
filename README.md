# The Official Implementation of STA

## 1. Download the Dataset

Download datasets from HuggingFace:
- [MOSEI](https://huggingface.co/datasets/AZYoung/MOSEI_processed)
- [SIMS-V2](https://huggingface.co/datasets/AZYoung/SIMSV2_processed)
- [MELD](https://huggingface.co/datasets/AZYoung/MELD_processed)
- [CHERMA](https://huggingface.co/datasets/AZYoung/CHERMA0723_processed)

Place them under the same folder, and set `root_dataset_dir` in `parse_args` of `run.py` to the dataset path.

## 2. Download the Backbone LLM

Download [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) and set `pretrain_LM` in `parse_args` of `run.py` to the LLM path.

> If download is too slow, try [Modelscope](https://modelscope.cn/) or HF-mirrors(https://hf-mirror.com/).

## 3. Acknowledgment

Our code is structurally referenced to [MSE-Adapter](https://github.com/AZYoung233/MSE-Adapter) and [Self-MM](https://github.com/thuiar/Self-MM). Thanks for their open-source spirit!
