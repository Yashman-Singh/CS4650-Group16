# Sentiment Analysis with Emojis — Dual-Stream Model

## Project Overview

This project investigates how incorporating emojis into sentiment analysis models can impact performance compared to traditional text-only approaches.  
We implement a dual-stream architecture that combines:

- **Text features** extracted using a fine-tuned BERT model
- **Emoji features** extracted using pre-trained Emoji2Vec embeddings

Both streams are fused using a cross-modal attention mechanism to classify the sentiment of tweets into **positive** or **negative** classes.

A full ablation study is conducted comparing:
- Text-Only Model
- Dual-Stream (Text + Emoji) Model

---

## Repository Structure

- `preprocess.ipynb` — Preprocessing script to clean and prepare the Sentiment140 dataset, including emoticon-to-emoji conversion.
- `main.ipynb` — Final training and evaluation notebook for the Dual-Stream and Text-Only models.
- `data/` — Directory containing the processed CSV and pre-trained Emoji2Vec embeddings. (The original raw dataset is too large for GitHub but can be found at the mentioned link)

---

## Dataset

The original dataset used is [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140), available on Kaggle.  
- Raw data was preprocessed to replace ASCII-style emoticons (e.g., `:)`, `:(`) with their Unicode emoji equivalents (e.g., 🙂, 🙁).
- The final sampled dataset (`sentiment140_with_emojis.csv`) contains 50,000 tweets with a balanced distribution of sentiment labels.

> **Dataset link:** [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## Instructions to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Yashman-Singh/CS4650-Group16.git
   cd CS4650-Group16

2. Download the original dataset from [here](https://www.kaggle.com/datasets/kazanova/sentiment140), and place the csv in the `data/` directory (ideally without changing file names)

3. Read/Run the `main.py` and `preprocess.py` files
