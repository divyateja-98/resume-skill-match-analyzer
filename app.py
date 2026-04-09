import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Resume Skill Match Analyzer", layout="wide")

st.title("🤖 Resume Skill Match & Job Fit Analyzer")

# --------------------
# Load Dataset
# --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/ai_job_dataset.csv")
    df["skill_list"] = df["required_skills"].apply(
        lambda x: [s.strip().lower() for s in str(x).split(",")]
    )
    return df

df = load_data()

# --------------------
# Skill Vectorization
# --------------------
vectorizer = CountVectorizer()
skill_matrix = vectorizer.fit_transform(df["required_skills"])

# --------------------
# Train ML model
# --------------------
encoder = LabelEncoder()
df["role_encoded"] = encoder.fit_transform(df["job_title"])

model = RandomForestClassifier(random_state=42)
model.fit(skill_matrix, df["role_encoded"])

# --------------------
# Sidebar Inputs
# --------------------
st.sidebar.header("Enter Candidate Skills")

user_skills = st.sidebar.text_input(
    "Enter skills separated by comma",
    "python, sql, pandas"
)

target_role = st.sidebar.selectbox(
    "Select Target Role",
    df["job_title"].unique()
)

# --------------------
# Prediction
# --------------------
if st.sidebar.button("Analyze Resume"):

    user_skills_clean = ",".join(
        [s.strip().lower() for s in user_skills.split(",")]
    )

    user_vector = vectorizer.transform([user_skills_clean])

    similarity_scores = cosine_similarity(
        user_vector,
        skill_matrix
    )

    predicted_role = encoder.inverse_transform(
        model.predict(user_vector)
    )[0]

    role_skills = set(
        df[df["job_title"] == target_role]
        ["skill_list"]
        .explode()
    )

    user_skill_set = set(user_skills_clean.split(","))

    skill_gap = role_skills - user_skill_set

    match_score = round(similarity_scores.max() * 100, 2)

    # --------------------
    # Results
    # --------------------
    st.subheader("📊 Results")

    st.write("Predicted Suitable Role:", predicted_role)

    st.write("Skill Match Score:", match_score, "%")

    st.subheader("🎯 Required Skills for Selected Role")
    st.write(role_skills)

    st.subheader("📉 Missing Skills")
    st.write(skill_gap)

    st.subheader("📚 Recommended Learning Path")

    for skill in skill_gap:
        st.write("•", skill)

# --------------------
# Dataset preview
# --------------------
st.subheader("Dataset Preview")
st.dataframe(df)
