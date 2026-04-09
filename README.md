# Resume Skill Match & Job Fit Analyzer

An end-to-end Machine Learning project that analyzes candidate skills and compares them with job requirements to predict job fit score, identify missing skills, and recommend learning priorities.

The system uses Natural Language Processing (NLP) techniques and similarity algorithms to evaluate how well a candidate’s skills match real job market requirements.

This project demonstrates practical Machine Learning workflow skills required for entry-level AI/ML roles.

---

# Problem Statement

Recruiters receive thousands of resumes for AI and Data Science roles.

Key challenges include:

- Identifying relevant candidates efficiently
- Matching candidate skills with job requirements
- Understanding skill gaps
- Providing structured learning guidance
- Reducing manual screening effort

This project builds an intelligent system that automates job-fit analysis using Machine Learning and NLP techniques.

---

# Objectives

- Extract skills from candidate input
- Compare candidate skills with job role requirements
- Calculate job match score
- Predict suitable job role
- Identify missing skills
- Recommend learning path

---

# Dataset

The dataset contains 50 entry-level AI job roles and their required skills.

Dataset features:

- job_title
- required_skills
- experience_level
- industry
- salary_usd

The dataset represents common skill combinations found in:

- Machine Learning roles
- Data Science roles
- AI roles
- Data Analyst roles
- MLOps roles

---

# Project Architecture

resume-skill-match-analyzer
│
├── data/
│   └── ai_job_dataset.csv
│
├── app.py
├── requirements.txt
└── README.md

---

# System Workflow

User Skills Input
↓
Data Cleaning
↓
Skill Vectorization (CountVectorizer)
↓
Similarity Calculation (Cosine Similarity)
↓
Machine Learning Model (RandomForest)
↓
Role Prediction
↓
Skill Gap Identification
↓
Learning Recommendation
↓
Interactive Dashboard (Streamlit)

---

# Tech Stack

Python

Pandas

NumPy

Scikit-learn

Natural Language Processing (CountVectorizer)

Streamlit

---

# Machine Learning Approach

1. Text preprocessing of skills
2. Conversion of skills into numerical vectors using CountVectorizer
3. Cosine similarity used to compute skill match score
4. RandomForest Classifier used to predict suitable job role
5. Skill gap identified by comparing required skills with user skills

---

# Features

- Predict suitable job role
- Calculate skill match score
- Identify missing skills
- Recommend learning path
- Interactive Streamlit dashboard
- End-to-end ML pipeline
- Real-world use case

---

# How to Run the Project

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

---

# Example Input

User skills:

python, sql, pandas

System output:

Predicted Role

Skill Match Score

Missing Skills

Recommended Learning Path

---

# Learning Outcomes

- Data preprocessing
- Feature engineering
- NLP basics
- Machine Learning modeling
- Similarity algorithms
- Model deployment
- Streamlit dashboard development

---

# Future Improvements

- Resume PDF parser
- Use TF-IDF or word embeddings
- Integrate real-time job market API
- Add visualization dashboards
- Improve recommendation algorithm

---

# Conclusion

This project demonstrates the ability to build an end-to-end Machine Learning system that solves a real-world hiring problem using practical data science techniques.

The system helps understand how skills relate to job requirements and provides structured guidance for career improvement.
