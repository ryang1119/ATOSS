# **ATOSS: Aspect Term-Oriented Sentence Splitter

Repo for EMNLP 2024 Findings paper **📄 [Read the Paper: ATOSS](https://arxiv.org/abs/2410.02297)**  

## Quick Start

Follow the steps below ⬇️

### **1. Set Up Environment**

```sh
conda create -n atoss python=3.8
conda activate atoss
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

---

### **2. Train & Inference**

run with command:

```sh
cd src
```

#### **Run SFT (Supervised Fine-Tuning):**

```sh
sh run_sft.sh
```

#### **Run DPO (Direct Preference Optimization):**

```sh
sh run_dpo.sh
```

---

### **3. Evaluate Results**

Once the inference steps for SFT and DPO are completed, you can evaluate the results using baseline models:

- **Paraphrase**  
- **DLO & ILO**  
- **MVP**

Evaluation files:
- `sft_test.txt` (for SFT results)  
- `dpo_test.txt` (for DPO results)

---

### **4. Directory Structure**

Here’s an overview of the key directories in this project:

```
ATOSS/
├── src/
│   ├── run_sft.sh      # Script to run SFT
│   ├── run_dpo.sh      # Script to run DPO
│   └── ...             # Additional source files
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── ...
```

---

### **5. Citation**

If you find ATOSS helpful for your research, please cite:

```bibtex
@article{seo2024make,
  title={Make Compound Sentences Simple to Analyze: Learning to Split Sentences for Aspect-based Sentiment Analysis},
  author={Seo, Yongsik and Song, Sungwon and Heo, Ryang and Kim, Jieyong and Lee, Dongha},
  journal={arXiv preprint arXiv:2410.02297},
  year={2024}
}
```

---

### **6. Contact**

For questions, suggestions, or issues, feel free to reach out:

- **Email:** [ryang1119@yonsei.ac.kr](mailto:ryang1119@yonsei.ac.kr)

