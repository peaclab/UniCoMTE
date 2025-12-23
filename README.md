# UniCoMTE (CoMTE V1.1)
**Universal Counterfactual Explanations for Multivariate Time Series Classifiers**

UniCoMTE is a **model-agnostic explainability framework** designed to generate **counterfactual explanations** for predictions made by multivariate time series (MTS) classifiers.  
A counterfactual explanation answers the question:  
> *“What minimal change in this time series would have led the model to predict a different class?”*

Unlike the original [CoMTE](https://github.com/peaclab/CoMTE) framework, which was implemented for specific model backends, **UniCoMTE** introduces a **unified and extensible pipeline** that supports multiple machine learning ecosystems, including **scikit-learn**, **PyTorch**, and **TensorFlow**—all without requiring model-specific modifications.

---

## 🚀 Key Features
- 🧩 **Model-Agnostic:** Works with diverse ML frameworks via a user-defined wrapper interface.  
- ⚙️ **Model Wrapper:** Allows user to define commands for model queries, and pre and post-processing steps for black-box classifiers.
- ⚙️ **Data Wrapper:** Allows user to define operations to handle various input data formats.
- 🔍 **Efficient Distractor Retrieval:** Uses **class-specific KD-trees** for fast nearest-neighbor searches among correctly classified samples.  
- 📉 **Sparse, Actionable Explanations:** Identifies the **smallest set of variable–time pairs** that must change to alter classification.  
- 🧠 **Physiologically Relevant:** Validated on healthcare applications like **ECG diagnosis**, where temporal and variable dependencies are critical.  

---

## 🧱 Framework Overview

UniCoMTE consists of three modular components:

### 1. Data & Model Wrapper  
- Allows user to standardize data and model interfaces across ML backends.  
- For model wrapper, user can define operations for:  
  - Pre-processing (e.g., normalization, batching, reshaping)
  - Inference command
  - Post-processing (e.g., thresholding, calibration, Logit-to-class mapping)
  - Mode switching (e.g., eager, evaluation)
- For the data wrapper, the user can define operations for:
  - Sample transformations: converting custom training and testing samples into canonical representations (e.g., multi-index NumPy arrays)
  - Label transformations: converting custom labels into standardized NumPy arrays suitable for training and evaluation

### 2. Distractor Selection Module  
- Retrieves *distractors*—samples from the target class that closely resemble the instance being explained.  
- Employs **class-specific KD-trees** to efficiently find nearest neighbors in feature space.

### 3. Counterfactual Generation Module  
- Performs a **discrete random hill-climbing search** to identify minimal feature–time substitutions that flip the predicted class.  
- Falls back to a **greedy incremental strategy** when necessary.  
- Produces counterfactuals that are sparse, interpretable, and faithful to the model’s decision boundary.

The resulting counterfactuals reveal **which waveform segments or variables are critical** to the model’s prediction and **how they must change** to yield a different outcome.


## ⚙️ Installation

### Requirements
- Python 3.8+
- All dependencies listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📄 Citation

If you use **UniCoMTE** in your research, please cite our paper:

**Justin Li, Efe Sencan, Jasper Zheng Duan, Vitus J. Leung, Stephen Tsaur, Ayşe K. Coskun.**  
*UniCoMTE: A Universal Counterfactual Framework for Explaining Time-Series Classifiers on ECG Data.*  
arXiv preprint, Dec. 2025.

🔗 https://arxiv.org/abs/2512.17100

### BibTeX
```bibtex
@article{li2025unicomte,
  title   = {UniCoMTE: A Universal Counterfactual Framework for Explaining Time-Series Classifiers on ECG Data},
  author  = {Li, Justin and Sencan, Efe and Duan, Jasper Zheng and Leung, Vitus J. and Tsaur, Stephen and Coskun, Ayse K.},
  journal = {arXiv preprint arXiv:2512.17100},
  year    = {2025}
}

