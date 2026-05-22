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
| 4 | Text Preprocessing, Conversational Context & TF-IDF |
| 5 | Classical ML — Hyperparameter Tuning (LogReg, LinearSVC) |
| 6 | Deep Learning — DeBERTa-v3-small (Transformers + AMP) |
| 7 | Model Comparison & Results |
| 8 | Inference Demo |

## 🚀 Cara Pakai

1. Buka [Kaggle Notebooks](https://www.kaggle.com/code)
2. Upload `satria-data-try.ipynb`
3. Add Input → cari **"Multimodal EmotionLines Dataset(MELD)"** by zaber666
4. Add Input → cari **"deberta-v3-small-starter-cv-0-820-lb-0-800"** by cdeotte
5. Enable **GPU** accelerator (P100 / T4 x2)
6. Run All

## ⚡ Optimasi

- **Conversational Context**: Menggabungkan histori kalimat dengan token `[SEP]`.
- **Offline Models**: DeBERTa dimuat dari local dataset tanpa butuh koneksi internet.
- **Mixed Precision (AMP)**: PyTorch `autocast` melatih model Deep Learning 2x lebih cepat dengan memori 50% lebih efisien.
- **Hyperparameter Tuning**: Otomatis mencari parameter ML terbaik dengan `RandomizedSearchCV` + TF-IDF Pipeline.
- **Efficient Loading**: `usecols` dan Dtype downcast menghemat ~30-50% RAM.
- **Class Weights**: Handling data imbalanced secara otomatis di Loss Function.

## 📊 Models

- **Logistic Regression (Tuned)** (Pipeline TF-IDF + class_weight)
- **LinearSVC (Tuned)** (Pipeline TF-IDF + class_weight)
- **Multinomial Naive Bayes** (Pipeline TF-IDF)
- **Random Forest** (Pipeline TF-IDF + class_weight)
- **Gradient Boosting** (Pipeline TF-IDF)
- **DeBERTa-v3-small** (Hugging Face Transformers + PyTorch AMP)

## 📄 License

Dataset: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)