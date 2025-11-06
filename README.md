# UniCoMTE (CoMTEv1.1)
**Universal Counterfactual Explanations for Multivariate Time Series Classifiers**

UniCoMTE is a **model-agnostic explainability framework** designed to generate **counterfactual explanations** for predictions made by multivariate time series (MTS) classifiers.  
A counterfactual explanation answers the question:  
> *“What minimal change in this time series would have led the model to predict a different class?”*

Unlike the original [CoMTE](https://www.bu.edu/peaclab/files/2021/05/CoMTE___ICAPAI.pdf) framework, which was implemented for specific model backends, **UniCoMTE** introduces a **unified and extensible pipeline** that supports multiple machine learning ecosystems, including **scikit-learn**, **PyTorch**, and **TensorFlow**—all without requiring model-specific modifications.

---

## 🚀 Key Features
- 🧩 **Model-Agnostic:** Works seamlessly with diverse ML frameworks via a standardized wrapper interface.  
- ⚙️ **Model Wrrapper:** Automatically handles model queries, preprocessing, and inference for black-box classifiers.
- ⚙️ **Data Wrrapper:** Automatically handles various data formats to ensure allignment.
- 🔍 **Efficient Distractor Retrieval:** Uses **class-specific KD-trees** for fast nearest-neighbor searches among correctly classified samples.  
- 📉 **Sparse, Actionable Explanations:** Identifies the **smallest set of variable–time pairs** that must change to alter classification.  
- 🧠 **Physiologically Relevant:** Validated on healthcare applications like **ECG diagnosis**, where temporal and variable dependencies are critical.  

---

## 🧱 Framework Overview

UniCoMTE consists of three modular components:

### 1. Data & Model Wrapper  
- Standardizes data and model interfaces across ML backends.  
- Automatically handles:  
  - Preprocessing (e.g., normalization, batching, reshaping)  
  - Mode switching (eager/evaluation)  
  - Logit-to-class mapping and probability queries  
- Supports custom post-processing (e.g., thresholding or calibration).

### 2. Distractor Selection Module  
- Retrieves *distractors*—samples from the target class that closely resemble the instance being explained.  
- Employs **class-specific KD-trees** to efficiently find nearest neighbors in feature space.

### 3. Counterfactual Generation Module  
- Performs a **discrete random hill-climbing search** to identify minimal feature–time substitutions that flip the predicted class.  
- Falls back to a **greedy incremental strategy** when necessary.  
- Produces counterfactuals that are sparse, interpretable, and faithful to the model’s decision boundary.

The resulting counterfactuals reveal **which waveform segments or variables are critical** to the model’s prediction and **how they must change** to yield a different outcome.

---

## ⚙️ Installation

### Requirements
- Python 3.8+
- All dependencies listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
