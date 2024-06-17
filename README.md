# ATOSS

## Quick Start

First, transfer the data folder into the ATOSS folder.

Follow the steps below ⬇️

### Set Up

```sh
conda create -n atoss python=3.8
conda activate atoss
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### Train & Inference

run with command:

cd src

if you want to run SFT:

```sh
sh run_sft.sh
```

if you want to run DPO:

```sh
sh run_dpo.sh
```

### After inference of SFT and DPO is completed..

you can inference sft_test.txt and dpo_test.txt to the baseline model such as mvp and paraphrase.
