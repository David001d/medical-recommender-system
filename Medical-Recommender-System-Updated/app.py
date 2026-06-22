import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Medical Specialist Recommender",
    page_icon="🩺",
    layout="centered"
)

# ==========================
# CUSTOM FRONTEND DESIGN
# ==========================

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #F5F7FA;
        }

        .block-container {
            padding-top: 0rem;
            max-width: 850px;
        }

        .main-header {
            background-color: #2A6EBB;
            padding: 24px 28px;
            border-radius: 0 0 22px 22px;
            margin-bottom: 28px;
            box-shadow: 0 8px 20px rgba(42, 110, 187, 0.18);
        }

        .header-row {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .header-icon {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            background: rgba(255,255,255,0.16);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 27px;
        }

        .main-title {
            color: white;
            font-size: 28px;
            font-weight: 700;
            margin: 0;
            line-height: 1.15;
        }

        .main-subtitle {
            color: #A8C8E8;
            font-size: 14px;
            margin-top: 5px;
        }

        .form-card {
            background: white;
            padding: 30px;
            border-radius: 22px;
            box-shadow: 0 10px 24px rgba(26, 43, 74, 0.08);
            border: 1px solid #E8EDF2;
            margin-bottom: 22px;
        }

        .section-title {
            color: #1A2B4A;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .section-subtitle {
            color: #4A5568;
            font-size: 14px;
            margin-bottom: 22px;
        }

        label, .stTextInput label, .stTextArea label, .stSelectbox label {
            color: #1A2B4A !important;
            font-size: 14px !important;
            font-weight: 600 !important;
        }

        textarea, input, select {
            border-radius: 11px !important;
            border: 1px solid #D1D9E0 !important;
            color: #1A2B4A !important;
        }

        textarea:focus, input:focus {
            border-color: #2A6EBB !important;
            box-shadow: 0 0 0 2px rgba(42, 110, 187, 0.12) !important;
        }

        div.stButton > button:first-child {
            width: 100%;
            height: 50px;
            background-color: #2A6EBB;
            color: white;
            border-radius: 14px;
            border: none;
            font-weight: 700;
            font-size: 15px;
            letter-spacing: 0.01em;
            margin-top: 8px;
            transition: 0.2s ease;
        }

        div.stButton > button:first-child:hover {
            background-color: #1E5A9E;
            color: white;
            border: none;
            transform: translateY(-1px);
        }

        .result-card {
            background-color: #EDF7F0;
            border-radius: 22px;
            padding: 26px;
            border: 1px solid #C8EDD7;
            margin-top: 22px;
            margin-bottom: 14px;
        }

        .result-row {
            display: flex;
            align-items: flex-start;
            gap: 14px;
        }

        .result-icon {
            color: #27AE60;
            font-size: 25px;
            margin-top: 1px;
        }

        .result-title {
            color: #1A2B4A;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 16px;
            line-height: 1.2;
        }

        .confidence-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #4A5568;
            font-size: 13px;
            margin-bottom: 7px;
        }

        .confidence-value {
            color: #27AE60;
            font-weight: 700;
        }

        .progress-bg {
            width: 100%;
            height: 8px;
            background-color: #C8EDD7;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 14px;
        }

        .progress-fill {
            height: 8px;
            background-color: #27AE60;
            border-radius: 999px;
        }

        .matched-text {
            color: #4A5568;
            font-size: 13px;
            line-height: 1.5;
        }

        .matched-text strong {
            color: #1A2B4A;
        }

        .warning-card {
            background-color: #FFFBEA;
            border: 1px solid #F6E58D;
            border-radius: 16px;
            padding: 16px 18px;
            display: flex;
            gap: 12px;
            align-items: flex-start;
            color: #7D5A00;
            font-size: 13px;
            line-height: 1.5;
            margin-bottom: 22px;
        }

        .dataset-card {
            background: white;
            border-radius: 20px;
            padding: 8px 18px 18px 18px;
            box-shadow: 0 6px 18px rgba(26, 43, 74, 0.05);
            border: 1px solid #E8EDF2;
            margin-top: 18px;
        }

        .dataset-title {
            color: #1A2B4A;
            font-size: 15px;
            font-weight: 700;
            padding-top: 10px;
            margin-bottom: 10px;
        }

        .footer-note {
            text-align: center;
            color: #718096;
            font-size: 12px;
            margin-top: 25px;
        }

        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# HEADER
# ==========================

st.markdown(
    """
    <div class="main-header">
        <div class="header-row">
            <div class="header-icon">🩺</div>
            <div>
                <h1 class="main-title">Medical Specialist Recommender</h1>
                <div class="main-subtitle">Symptom-based specialist matching</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================
# LOAD DATASET
# ==========================

@st.cache_data
def load_data():
    return pd.read_csv("medical_dataset.csv")

df = load_data()

features = [
    "symptoms",
    "severity",
    "duration",
    "body_part",
    "condition_type"
]

for feature in features:
    df[feature] = df[feature].fillna("")

df["combined_features"] = (
    df["symptoms"] + " " +
    df["severity"] + " " +
    df["duration"] + " " +
    df["body_part"] + " " +
    df["condition_type"]
)

# ==========================
# MODEL
# ==========================

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

def recommend_specialist(user_input):
    user_vector = cv.transform([user_input.lower()])
    similarity_scores = cosine_similarity(user_vector, count_matrix)

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.1:
        return "General Physician", best_score, "No strong match found"

    specialist = df.iloc[best_match_index]["specialist"]
    matched_symptoms = df.iloc[best_match_index]["symptoms"]

    return specialist, best_score, matched_symptoms

# ==========================
# FORM CARD
# ==========================

st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Describe your symptoms and receive a suggested specialist.</div>',
    unsafe_allow_html=True
)

with st.form("recommendation_form"):
    symptoms = st.text_area(
        "Describe your symptoms",
        placeholder="e.g. chest pain and shortness of breath",
        height=105
    )

    severity = st.selectbox(
        "Severity",
        ["Mild", "Moderate", "Severe"],
        index=1
    )

    duration = st.text_input(
        "How long have you had these symptoms?",
        placeholder="e.g. 3 days"
    )

    body_part = st.text_input(
        "Affected body part",
        placeholder="e.g. chest"
    )

    submitted = st.form_submit_button("Find My Specialist")

 

# ==========================
# RESULT CARD
# ==========================

# ==========================
# RESULT CARD
# ==========================

if submitted:
    if not symptoms.strip():
        st.error("Please enter your symptoms.")
    else:
        user_input = f"{symptoms} {severity} {duration} {body_part}"

        specialist, score, matched_case = recommend_specialist(user_input)
        confidence = round(score * 100, 2)

        st.success(f"✅ Recommended Specialist: {specialist}")

        st.write("**Match Confidence:**", f"{confidence}%")
        st.progress(min(score, 1.0))

        st.write("**Matched Symptoms:**", matched_case)

        

# ==========================
# DATASET SECTION
# ==========================

 
st.markdown('<div class="dataset-title">📊 View Dataset</div>', unsafe_allow_html=True)

with st.expander("Open medical dataset"):
    st.dataframe(
        df[["symptoms", "severity", "duration", "body_part", "condition_type", "specialist"]],
        use_container_width=True
    )



