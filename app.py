import streamlit as st
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('fake_job_posting_model.joblib')
    return model

model = load_model()

# App Title and Description
st.title("🛡️ Fake Job Posting Detector")
st.write("This machine learning app detects whether a job posting is **Real** or **Fraudulent** based on its text content.")

# Input Fields
st.subheader("Job Details")
title = st.text_input("Job Title")
location = st.text_input("Location")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description", height=150)
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

# Prediction Button
if st.button("Analyze Job Posting"):
    # Combine inputs into a single string (matching training data format)
    full_text = f"{title} {location} {company_profile} {description} {requirements} {benefits}"
    
    if full_text.strip():
        # Make Prediction
        prediction = model.predict([full_text])[0]
        probability = model.predict_proba([full_text])[0]

        # Display Result
        if prediction == 1:
            st.error("⚠️ **FRAUDULENT JOB DETECTED**")
            st.write(f"Confidence: {probability[1]*100:.2f}%")
        else:
            st.success("✅ **REAL JOB**")
            st.write(f"Confidence: {probability[0]*100:.2f}%")
    else:
        st.warning("Please enter some details to analyze.")
      
