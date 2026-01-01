import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Dataset
data_file = 'fake_job_postings.csv'
df = pd.read_csv(data_file)

# 2. Preprocessing
# Fill missing values with empty strings
text_columns = ['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    df[col] = df[col].fillna('')

# Combine all text columns into a single feature
df['text'] = df[text_columns].apply(lambda x: ' '.join(x), axis=1)

# Split into Features (X) and Target (y)
X = df['text']
y = df['fraudulent']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Build Pipeline (Vectorization + Classifier)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 4. Train Model
print("Training model...")
pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = pipeline.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Save the Model
joblib.dump(pipeline, 'fake_job_posting_model.joblib')
print("Model saved as 'fake_job_posting_model.joblib'")
