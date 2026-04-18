# 🛡️ Deep Learning for Multi-Label Content Moderation

**Author:** Dilraj Brar | B.Tech CSE Core @ SRM Institute of Science and Technology  
**Domain:** Natural Language Processing (NLP) & Deep Learning  

---

## 📌 Project Overview
This project engineers an automated content moderation system capable of precisely identifying overlapping categories of online harassment in a single inference pass. Unlike standard multi-class classification, this model solves a **Multi-Label** problem, determining independent probabilities for six simultaneous toxicity categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

## ⚙️ The Core Engine: Bidirectional LSTM
Traditional baseline models (Logistic Regression, SVM) were built using TF-IDF vectorization, which ignores word order and semantic context. To achieve state-of-the-art accuracy, a deep recurrent neural network was developed.

**Architecture Details:**
* **Embedding Layer:** 128-dimensional dense word vectors mapped from a 20,000-word vocabulary limit.
* **Bi-LSTM Layer:** 60 units processing 150-word padded sequences in both forward and backward directions. This bidirectional context retention is critical for capturing sarcasm, negations, and complex insults.
* **Feature Extraction & Regularization:** Global 1D Max Pooling combined with Dropout (0.1) to prevent overfitting.
* **Output Layer:** 6 independent neurons utilizing a **Sigmoid** activation function to output distinct probabilities (0.0 to 1.0) for all six categories simultaneously.
* **Total Trainable Parameters:** ~2.8 Million.

## 📊 Results & Comparative Analysis

| Model | Architecture Type | Macro F1-Score | Overall ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Baseline (TF-IDF) | 0.4682 | 0.9080 |
| **Linear SVC (SVM)** | Baseline (TF-IDF) | 0.6074 | N/A |
| **Bidirectional LSTM** | **Deep Learning (Proposed)** | **0.6358** | **0.9766 🏆** |

**Key Finding:** The Bi-LSTM outperformed statistical baselines in every metric, achieving near-perfect ranking accuracy (**0.97+ ROC-AUC**) across extremely rare toxicity classes (e.g., 0.985 for `severe_toxic` and 0.981 for `threat`), proving the superiority of bidirectional context retention.

## ⚖️ Overcoming Extreme Class Imbalance
The dataset presents a massive class imbalance (e.g., ~143,000 clean comments vs. only 478 'threat' comments). Rather than artificially distorting the natural language data via oversampling, this challenge was mitigated through:
1. **Evaluation Strategy:** Utilizing Macro-Averaged F1 and ROC-AUC scores to ensure rare classes were weighted equally to majority classes during evaluation.
2. **Semantic Generalization:** Leveraging Word Embeddings to group similar concepts mathematically, allowing the Bi-LSTM to understand the *intent* of rare toxic traits far better than frequency-based baseline models.

## 🗄️ Dataset & Provenance
The model was trained and evaluated using the **[Jigsaw Toxic Comment Classification Challenge Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)**, sourced via Kaggle. 

*Note: Due to GitHub's file size constraints, the raw 153,000+ row dataset (`train.csv`) is not hosted directly in this repository. Researchers and developers can access the raw data via the Kaggle link above.*
## 📂 Repository Structure
* `/notebooks/BiLSTM_Content_Moderation.ipynb`: The complete Google Colab Jupyter Notebook containing the data pipeline, model training, and evaluation code.
* `/visuals/`: Directory containing project visualizations including the Model Performance Comparison chart, Dataset Distribution analysis, and the final academic presentation poster.
* **Pre-trained Model:** Due to GitHub file size limits, the final `.h5` model weights are hosted in the **[Releases](../../releases)** section of this repository.

## 🚀 Usage & Inference
To load the pre-trained engine into a local Python environment without retraining:

1. Download the `toxic_comment_bilstm.h5` file from the **[Releases](../../releases)** page.
2. Load it into your environment using TensorFlow/Keras:

```python
from tensorflow.keras.models import load_model

# Load the trained engine
moderation_engine = load_model('toxic_comment_bilstm.h5')

# Ensure your input text is processed via the saved Tokenizer first
# predictions = moderation_engine.predict(padded_sequences)
