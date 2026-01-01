# Fake-Job-Posting-Detection
# 🛡️ Fake Job Posting Detection

A machine learning web application that detects whether a job posting is **Real** or **Fraudulent** based on its text content (title, description, requirements, etc.). Built with Python, Scikit-Learn, and Streamlit, and ready for deployment on Render.

## 🚀 Features
- **Machine Learning Model**: Uses a Random Forest Classifier with TF-IDF vectorization.
- **Interactive UI**: User-friendly web interface built with Streamlit.
- **Real-time Prediction**: Instantly analyzes job details and provides a confidence score.
- **Deployment Ready**: Configured for easy hosting on Render.

## 📂 Project Structure
```text
fake-job-detector/
├── app.py                         # The Streamlit web application
├── train_model.py                 # Script to train and save the ML model
├── fake_job_postings.csv          # The dataset file
├── fake_job_posting_model.joblib  # The saved model (generated after training)
├── requirements.txt               # List of python dependencies
└── README.md                      # Project documentation
