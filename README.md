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


## ⚙️ Installation and Usage

### Requirements
- Python 3.8+
- All dependencies listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
```
https://github.com/peaclab/UniCoMTE/blob/e3afbab7c052024260b4f48013a7187064c74fd1/UniCoMTE/Full_ECG_Data_Preprocessing_Pipeline.ipynb

### Usage

Using UniCoMTE with a different model and dataset: 
1. Create a copy of the data and model wrappers
2. Edit the data wrapper to make necessary transformations. The end state of the data is listed in the data wrapper.
3. Define the model wrapper in the script or jupyter notebook.
4. Import data and model wrappers into the environment where UniCoMTE is being run.


Our Implementation:

“[UniCoMTE/Sandia_Comlex_ECG_Implementation_Partial_Training.ipynb]([url](https://github.com/peaclab/UniCoMTE/blob/main/UniCoMTE/Sandia_Comlex_ECG_Implementation_Partial_Training.ipynb))” demonstrates the application of the official version of UniCoMTE that utilizes a subset of the CODE15 training dataset. 

“[UniCoMTE/Sandia_Comlex_ECG_Implementation_Full_Training.ipynb]([url](https://github.com/peaclab/UniCoMTE/blob/main/UniCoMTE/Sandia_Comlex_ECG_Implementation_Full_Training.ipynb))” demonstrates the application of a modified version of UniCoMTE that utilizes the full CODE15 training dataset. 

“[UniCoMTE/Full_ECG_Data_Preprocessing_Pipeline.ipynb]([url](https://github.com/peaclab/UniCoMTE/blob/main/UniCoMTE/Full_ECG_Data_Preprocessing_Pipeline.ipynb))” demonstrates the extraction and preprocessing of the CODE testing and training data.

“[UniCoMTE/comlex_core/src/explainable_data_ECG.py]([url](https://github.com/peaclab/UniCoMTE/blob/main/UniCoMTE/comlex_core/src/explainable_data_ECG.py))” and “[UniCoMTE/comlex_core/src/explainable_model_ECG.py](UniCoMTE/comlex_core/src/explainable_model_ECG.py)” contain the data and model wrappers, respectively, used in our implementation. The user can define these according to the format of their data and classification algorithms.

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

