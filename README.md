# SpamDetector

This repository provides a complete solution for detecting spam emails in both English and Persian (Farsi). It includes preprocessing pipelines, model architecture, training scripts, FastText embeddings, and evaluation tools.

## Overview

SpamDetector is a deep learning-based binary classifier trained to distinguish between **spam** and **ham** (non-spam) emails. It leverages a hybrid neural network architecture combining Convolutional Layers, Bi-LSTM, and Attention mechanism to learn from multilingual email data. The model uses pre-trained FastText embeddings for both English and Persian languages.

## Features

- Multilingual support (English and Persian)
- Combined Conv1D + BiLSTM + Attention model
- Pre-trained FastText embeddings (300-dim)
- Custom data loader with encoding detection
- Stratified train/test split and validation
- F1, Precision, Recall, AUC metrics
- Sample evaluation with real-world messages
- Compatible with GPU (CUDA memory growth configured)

## Dataset

The training data is composed of 5 diverse datasets combined and cleaned, including:
- Public spam email datasets
- Manually curated Persian spam and ham emails
- Unified and shuffled into a single `emails.csv`

## Model Architecture

```
Input → Embedding (FastText)
      → Conv1D + MaxPooling + Dropout
      → BiLSTM (return_sequences=True)
      → Attention + GlobalMaxPooling
      → Dense + BatchNorm + Dropout
      → Output (Sigmoid)
```

## Requirements

- Python 3.11
- TensorFlow ≥ 2.12
- Keras
- NumPy, pandas, seaborn, scikit-learn
- FastText vectors for EN and FA
- Jupyter or Colab for running the notebook

## Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/masoumehkhaleghian/SpamDetector.git
   cd SpamDetector
   ```

2. Organize your files:
   - Store your combined dataset at: `Dataset/emails.csv`
   - Place FastText `.vec` files at:
     ```
     FastText/
     ├── EN/cc.en.300.vec
     └── FA/cc.fa.300.vec
     ```
   - Save best model weights under: `Checkpoint/model_checkpoint.h5`

3. Run training:
   Use the Colab notebook or Python script to train your model. Early stopping and checkpointing are already integrated.

4. Run predictions:
   Use the utility function `prepare_spam_data()` to preprocess your text and feed it into the model:
   ```python
   X, y = prepare_spam_data(sample_emails, sample_labels, tokenizer)
   y_pred = (model.predict(X) > 0.5).astype("int32")
   ```

## Evaluation

Model evaluation includes:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix visualization

These metrics are computed on both test datasets and custom sample inputs.

## Results

| Metric     | Value (Test Set) |
|------------|------------------|
| Accuracy   | > 95%            |
| F1-Score   | ~0.96            |
| AUC        | ~0.98            |

## File Structure

```
SpamDetector/
├── Dataset/
│   └── emails.csv
├── FastText/
│   ├── EN/cc.en.300.vec
│   └── FA/cc.fa.300.vec
├── Checkpoint/
│   └── model_checkpoint.h5
├── spamdetector.py
└── SpamDetector.ipynb
```

## Assets

All supporting files, including:
- Trained model weights
- Embeddings
- Datasets
- Source code
- Visualization notebooks

are also available in this shared Google Drive folder:

📁 [Access Full Project Files](https://drive.google.com/drive/folders/1BJVMNyYuNi48djdcKaFkHtaOwjlfx1t7?usp=sharing)

---

### Contact

For questions, please reach out via GitHub Issues or email listed in the repo.
