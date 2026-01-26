# 🧠 ASDFormer

A deep learning framework for fMRI analysis using the **ASDFormer** model.
This repository provides an environment setup guide and instructions to reproduce experiments.

This project is part of our paper:

Izadi, M., & Safayani, M. (2025). *ASDFormer: A Transformer with Mixtures of Pooling-Classifier Experts for Robust Autism Diagnosis and Biomarker Discovery*. arXiv:2508.14005. [Link](https://arxiv.org/abs/2508.14005)

---

## 🚀 Environment Setup

We recommend using **Conda** to manage dependencies.

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ASDFormer.git
   cd ASDFormer
   ```

2. **Create the Conda environment**

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**

   ```bash
   conda activate ASDFormer
   ```

---

## 📦 environment.yml

Below is the environment specification used for this project:

```yaml
name: ASDFormer
channels:
  - pytorch
  - defaults
  - conda-forge
dependencies:
  - python=3.9
  - cudatoolkit=11.3
  - pytorch=1.12.1
  - torchvision=0.13.1
  - torchaudio=0.12.1
  - scikit-learn=1.1.1
  - pandas=1.4.3
  - pip
  - pip:
      - wandb==0.13.1
      - hydra-core==1.2.0
```

---

## 🧪 Preparing the Dataset

Download the ABIDE dataset from [here](https://drive.google.com/file/d/14UGsikYH_SQ-d_GvY2Um2oEHw3WNxDY3/view?usp=sharing).
**After downloading, place the dataset in `source/dataset/` as `abide.npy`.**

---

## 🧪 Running the Model

Once the environment is ready and the dataset is in place, you can run the model using:

```bash
python -m source --multirun datasz=100p model=ASDFormer dataset=ABIDE repeat_time=5 preprocess=non_mixup
```

* `datasz=100p` → use 100% of the dataset
* `model=ASDFormer` → specify the model
* `dataset=ABIDE` → select dataset
* `repeat_time=5` → number of experiment repetitions
* `preprocess=non_mixup` → preprocessing strategy

---

## 📚 Inspirations & References

Our code was inspired by and builds upon the following projects:

* [Graphormer — Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/graphormer)
* [BrainNetworkTransformer — GitHub](https://github.com/Wayfear/BrainNetworkTransformer)

---

## 📖 Notes

* For GPU acceleration, ensure your machine has CUDA 11.3 drivers installed.
* Logs and experiment tracking are managed through [Weights & Biases (wandb)](https://wandb.ai/).

---

## 🏷 Citation

```bibtex
@misc{izadi2025asdformertransformermixturespoolingclassifier,
      title={ASDFormer: A Transformer with Mixtures of Pooling-Classifier Experts for Robust Autism Diagnosis and Biomarker Discovery}, 
      author={Mohammad Izadi and Mehran Safayani},
      year={2025},
      eprint={2508.14005},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.14005}, 
}
```
