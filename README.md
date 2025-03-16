# ZeLa: Advancing Zero-Shot Multilingual Semantic Parsing with Large Language Models and Chain-of-Thought Strategies

This repository contains the code and scripts for the *augmentation step* described in our paper, [**ZeLa**](https://aclanthology.org/2024.lrec-main.1547.pdf). After generating augmented training data with these scripts, you can fine-tune your semantic parser by following the instructions in [StructSP](https://github.com/truongdo619/StructSP).

## Overview

We provide two main steps for generating and processing augmented data:

1. **Step 1**: Generate raw augmented data using chain-of-thought (CoT) prompting.
2. **Step 2**: Extract decoupled forms from the raw augmented data to produce training-ready samples.

These scripts use large language models (e.g., Llama-2) to translate and augment text in multiple languages. After generating the augmented data and extracting the decoupled forms, you can then integrate the resulting dataset into your semantic parser training pipeline.

## Dependencies and Installation

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/truongdo619/ZeLa
   cd ZeLa
   ```

2. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Step-by-Step Usage

### 1. Generate Raw Augmented Data

Run the shell script for Step 1:
```bash
bash STEP_1_generate_raw_augmented_data.sh
```

### 2. Extract Decoupled Form

Run the shell script for Step 2:
```bash
bash STEP_2_extract_decoupled_form.sh
```

### 3. Fine-Tune Parser (Using StructSP)

After extracting decoupled forms, you can fine-tune the parser using [StructSP](https://github.com/truongdo619/StructSP).

## Citation

If you use this code, please cite our paper:
```
@inproceedings{do2024zela,
  title={ZeLa: Advancing Zero-Shot Multilingual Semantic Parsing with Large Language Models and Chain-of-Thought Strategies},
  author={Do, Truong Dinh and Nguyen, Phuong Minh and Nguyen, Minh},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={17783--17794},
  year={2024}
}
```

**Questions or issues?**  
Please open an Issue or contact us directly.
