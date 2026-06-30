# AI-Powered Automated Answer Sheet Evaluation System
**Bharat Academic CodeQuest 2026 | Team as5670252 | Akshat Singh**

---

## 🚀 Quick Setup (2 minutes)

### Prerequisites
- Python 3.8+
- Tesseract OCR installed on your system

### Install Tesseract
**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```
**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

**macOS:**
```bash
brew install tesseract
```

### Install Python dependencies
```bash
pip install -r requirements.txt
```

### Run the app
```bash
python app.py
```
Then open: **http://localhost:5000**

---

## 🧠 Pipeline

```
Upload Image → OpenCV Preprocess → Tesseract OCR → TF-IDF Scoring → Feedback → Store
```

1. **Preprocess** — Grayscale, denoise, adaptive threshold, deskew (OpenCV)
2. **OCR** — Extract handwritten text (Tesseract)
3. **Score** — TF-IDF cosine similarity + keyword matching (scikit-learn)
4. **Feedback** — Rule-based feedback with missing concept highlights
5. **Teacher Review** — Low-confidence answers flagged for human override

---

## 📁 Project Structure

```
answer_eval/
├── app.py              # Main Flask app + all pipeline logic
├── requirements.txt
├── uploads/            # Uploaded answer sheet images
└── templates/
    ├── index.html      # Main upload + evaluation UI
    ├── results.html    # Results table
    └── teacher.html    # Teacher dashboard (human-in-the-loop)
```

---

## 🎯 Demo Answer Key

The app comes preloaded with 3 sample questions:
- **Q1** — What is photosynthesis? (5 marks)
- **Q2** — Newton's First Law of Motion (5 marks)
- **Q3** — Pythagorean theorem (5 marks)

To test without a real handwritten sheet: write the answer on paper, photograph it, upload.

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Image Processing | Python + OpenCV |
| OCR | Tesseract 5.x |
| NLP Evaluation | TF-IDF + scikit-learn (cosine similarity) |
| Backend | Python Flask |
| Frontend | HTML + CSS (dark theme) |
| Storage | In-memory (prototype) |

> **Note:** Full version uses sentence-transformers for semantic NLP and MySQL for persistence. Prototype uses TF-IDF as a lightweight proxy.
