# SpamDetector

SpamDetector is a deep learning-based multilingual spam classification system. It leverages both Persian and English email datasets, FastText embeddings, and a hybrid LSTM-CNN architecture built using TensorFlow/Keras.

---

## Model Overview

The model is designed to classify emails as `spam` or `ham` based on their content. It combines:

- **Pretrained FastText Embeddings** for multilingual text understanding
- **1D Convolution + MaxPooling** for feature extraction
- **Bidirectional LSTM** for sequential modeling
- **Dropout layers** for regularization
- **Sigmoid output** for binary classification

---

## Project Structure

```

.
├── Dataset/                  # Raw and cleaned datasets
├── FastText/                 # Pretrained .vec embeddings
├── Checkpoint/               # Saved checkpoints (best per epoch)
├── Model/                    # Final trained model (.h5)
├── SpamDetector.ipynb        # Complete Jupyter notebook pipeline
├── README.md                 # Project documentation

````

---

## Dataset

Two datasets are used:

- `emails.csv`: Mixed-language dataset with `"email"` and binary `spam` labels (`0` = ham, `1` = spam)

---

## Preprocessing Pipeline

1. Remove punctuation from email content
2. Normalize labels (`0`/`1` → `ham`/`spam`)
3. Tokenize and convert to sequences
4. Pad all sequences to a max length of `400`
5. Split into train and test sets (80/20)

---

## Training Setup

- Embedding layer initialized with FastText
- Optimizer: `adam`
- Loss function: `binary_crossentropy`
- Epochs: 20 (with checkpoints)
- Max sequence length: 400
- Batch size: 128

Model performance is logged and best model is saved to `/Model`.

---

## Evaluation

The model is evaluated using:

- **Accuracy**
- **F1-score**
- **Confusion Matrix**
- **Classification Report (precision, recall)**

All metrics are visualized using `matplotlib` and `seaborn`.

---

## Inference

To make predictions on new messages:

```python
# Preprocess new emails
prepared_df, X, tokenizer, vocab_size = prepare_spam_data(sample_emails, sample_labels, tokenizer)

# Predict
y_pred_prob = model.predict(X)
y_pred = (y_pred_prob > 0.5).astype("int32")
````

Example messages (Persian + English) are also included in the notebook for real-world testing.

---

## Requirements

```bash
python >= 3.11
tensorflow >= 2.14
keras >= 2.14
pandas, numpy, seaborn
charset-normalizer
scikit-learn
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---
