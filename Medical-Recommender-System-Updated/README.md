# 🩺 Medical Specialist Recommendation System

A machine learning-based medical recommendation system that suggests the most suitable medical specialist based on a user's symptoms and related medical information.

This version includes a custom Streamlit frontend inspired by the provided design: blue medical header, clean white form card, green recommendation result card, caution banner, and dataset preview.

---

## 📌 Project Overview

The system compares user symptoms against a medical dataset and recommends the closest matching specialist using a content-based recommendation approach.

---

## 🎯 Features

- Symptom-based specialist recommendation
- CountVectorizer text vectorization
- Cosine similarity matching
- Confidence score calculation
- Custom Streamlit frontend design
- Dataset preview section
- Educational medical disclaimer

---

## 📂 Project Structure

```text
Medical-Recommender-System/
│
├── app.py
├── medical_dataset.csv
├── requirements.txt
├── README.md
│
└── assets/
    ├── home_page.png
    ├── recommendation_result.png
    ├── example_output.png
    └── dataset_preview.png
```

---

## 📊 Dataset Columns

| Column | Description |
|---|---|
| symptoms | Patient symptoms |
| severity | Severity level |
| duration | Duration of symptoms |
| body_part | Affected body part |
| condition_type | Type of medical condition |
| specialist | Recommended specialist |

---

## ⚙️ Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Default local URL:

```text
http://localhost:8501
```

---

## 🚀 Streamlit Cloud Deployment

1. Push this folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Deploy.

---

 
## 📜 License

This project is licensed under the MIT License.
