# resume-skill-match-analyzer
AI-powered system that analyzes resume skills, predicts job fit score, identifies skill gaps, and recommends learning paths using Machine Learning, NLP, and Streamlit.
# Resume Skill Match & Job Fit Analyzer

An ML-powered application that compares a candidate's skills with job requirements and predicts job fit score, identifies missing skills, and recommends learning priorities.

This project demonstrates practical skills required for entry-level ML/Data roles.

---

## Problem Statement

Recruiters receive thousands of resumes for AI/ML positions.

Challenges:

• Identifying relevant candidates quickly
• Matching resume skills with job requirements
• Understanding skill gaps
• Providing structured feedback

This project builds an intelligent system to automate job-fit analysis using machine learning and NLP techniques.

---

## Objectives

Extract skills from resume input
Compare with job role requirements
Calculate similarity score
Predict suitable job role
Identify missing skills
Recommend learning path

---

## Dataset

The dataset contains entry-level AI job roles and required skills.

Features:

job_title
required_skills
experience_level
industry
salary_usd

---

## System Workflow

User Skills Input
↓
Data Cleaning
↓
Skill Vectorization (CountVectorizer)
↓
Similarity Calculation (Cosine Similarity)
↓
ML Model Prediction (RandomForest)
↓
Skill Gap Detection
↓
Learning Recommendation
↓
Interactive Streamlit Dashboard

---

## Tech Stack

Python
Pandas
Scikit-learn
NLP (CountVectorizer)
Streamlit

---

## Features

Predict suitable job role
Calculate skill match score
Identify missing skills
Interactive dashboard
Simple ML pipeline
Real-world use case

---

## How to Run

pip install -r requirements.txt

streamlit run app.py

---

## Learning Outcomes

Data preprocessing
Feature engineering
NLP basics
Machine learning modeling
Similarity algorithms
Model deployment

---

## Future Improvements

PDF resume parser
Deep learning embeddings
Larger dataset
Real-time job scraping API
Advanced ranking algorithm

---

## Conclusion

This project demonstrates the ability to build an end-to-end ML system that solves a real-world hiring problem.
