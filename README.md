# 🧠 Satria Data 2025 — NLP Emotion Classification

Notebook untuk kompetisi **Satria Data 2025** menggunakan dataset **MELD** (Multimodal EmotionLines Dataset).

## 📋 Task

Klasifikasi emosi dari teks percakapan ke 7 kelas:
`anger` · `disgust` · `fear` · `joy` · `neutral` · `sadness` · `surprise`

## 📁 Dataset

[MELD Dataset](https://www.kaggle.com/datasets/zaber666/meld-dataset) — 13,000+ utterances dari serial TV *Friends*.

| Split | Utterances | Dialogues |
|-------|-----------|-----------|
| Train | ~9,989    | ~1,039    |
| Dev   | ~1,109    | ~114      |
| Test  | ~2,610    | ~280      |

## 🏗️ Pipeline (Notebook Sections)

| Section | Deskripsi |
|---------|-----------|
| 1 | Setup Environment & Config |
| 2 | Data Loading (optimized `usecols` + dtype downcast) |
| 3 | EDA — distribusi emosi, word cloud, panjang teks |
| 4 | Text Preprocessing & TF-IDF Feature Engineering |
| 5 | Classical ML — LogReg, LinearSVC, NB, RF, GBT |
| 6 | Deep Learning — BiLSTM with Attention |
| 7 | Model Comparison & Results |
| 8 | Inference Demo |

## 🚀 Cara Pakai

1. Buka [Kaggle Notebooks](https://www.kaggle.com/code)
2. Upload `satria-data-try.ipynb`
3. Add Input → cari **"Multimodal EmotionLines Dataset(MELD)"** by zaber666
4. Enable **GPU** accelerator
5. Run All

## ⚡ Optimasi

- Tidak ada `pip install` — semua library sudah ada di Kaggle
- Direct CSV path (bukan `os.walk` scan 11GB)
- `usecols` — hanya load kolom teks/emosi
- Dtype downcast — hemat ~30-50% RAM
- Matplotlib/Seaborn saja (tanpa Plotly ~200MB)
- Class weights untuk handle imbalanced data

## 📊 Models

- **Logistic Regression** (TF-IDF + class_weight)
- **LinearSVC** (TF-IDF + class_weight)
- **Multinomial Naive Bayes** (TF-IDF)
- **Random Forest** (TF-IDF + class_weight)
- **Gradient Boosting** (TF-IDF)
- **BiLSTM + Attention** (PyTorch, GPU)

## 📄 License

Dataset: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)