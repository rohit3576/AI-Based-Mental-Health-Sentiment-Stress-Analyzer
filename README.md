🧠 AI-Based Mental Health Sentiment & Stress Analyzer

🚀 Live Demo
🌐 Live App:
👉 https://ai-based-mental-health-sentiment-stress.onrender.com

📌 Project Overview

Mental health awareness is critical in today’s fast-paced world.
This project uses Natural Language Processing (NLP) and Deep Learning to analyze a user’s thoughts and emotions expressed through text.

The system:

✅ Detects sentiment (Positive / Negative)

✅ Estimates stress level (Low / Medium / High)

✅ Displays confidence score

✅ Provides wellness suggestions

✅ Features a modern glassmorphism UI

⚠️ Disclaimer: This tool is for educational purposes only and is not a medical diagnosis.

✨ Key Features

🧠 AI-powered Sentiment Analysis using Bidirectional LSTM

📊 Stress Level Classification (Low / Medium / High)

🎯 Confidence percentage visualization

🎨 Modern Glassmorphism UI (Mobile & Desktop responsive)

⏳ Loading animation for better UX

☁️ Free cloud deployment using Render

🛠️ Tech Stack
🔹 Backend

Python

Flask

TensorFlow / Keras

🔹 Machine Learning

Bidirectional LSTM (BiLSTM)

IMDB Dataset (Sentiment Learning)

Text padding & sequence modeling

🔹 Frontend

HTML5

CSS3 (Glassmorphism design)

Responsive UI

🔹 Deployment

GitHub

Render (Free Tier)

🧠 Model Architecture
Text Input
   ↓
IMDB Encoded Sequences
   ↓
Padding
   ↓
Embedding Layer
   ↓
Bidirectional LSTM
   ↓
Dense Layer (Sigmoid)
   ↓
Sentiment Score
   ↓
Stress Level + Suggestions

📊 Model Evaluation & Results
🔹 Sentiment Analysis (Binary Classification)

Metrics Used

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

ROC–AUC Curve

Confusion Matrix


ROC Curve


📁 Auto-generated evaluation report:
evaluation/evaluation_report_sentiment.csv

🔹 Stress Detection (Multiclass Classification)

Metrics Used

Accuracy

Precision (Weighted)

Recall (Weighted)

F1-Score (Weighted)

Confusion Matrix

Confusion Matrix


📁 Auto-generated evaluation report:
evaluation/evaluation_report_stress.csv

📌 Stress model is trained using simulated sentiment-score distributions for demonstration purposes.

📂 Project Structure
AI-Based-Mental-Health-Sentiment-Stress-Analyzer/
│
├── app.py
├── requirements.txt
│
├── model/
│   ├── sentiment_model.h5
│   └── stress_model.h5
│
├── evaluation/
│   ├── evaluation_sentiment.py
│   ├── evaluation_stress.py
│   ├── evaluation_report_sentiment.csv
│   ├── evaluation_report_stress.csv
│   └── plots/
│       ├── sentiment_confusion_matrix.png
│       ├── sentiment_roc_curve.png
│       └── stress_confusion_matrix.png
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md

🧪 Sample Input
I feel anxious and overwhelmed with my workload.

🔍 Output

Sentiment: Negative 😔

Stress Level: High Stress 😟

Confidence: 82%

Suggestion: Consider rest, talking to someone, or mindfulness.

▶️ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/rohit3576/AI-Based-Mental-Health-Sentiment-Stress-Analyzer.git
cd AI-Based-Mental-Health-Sentiment-Stress-Analyzer

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
python app.py


Open in browser:

http://127.0.0.1:5000

☁️ Deployment

This project is deployed on Render (Free Tier) using:

pip install -r requirements.txt
python app.py


✔ No paid services required
✔ Fully cloud hosted

🎓 Academic & Interview Relevance

This project demonstrates:

NLP preprocessing & sequence modeling

Deep learning with LSTM

Model evaluation (ROC, Confusion Matrix, F1-score)

Flask backend integration

UI/UX design

Free cloud deployment

🎯 Perfect for:

College final-year project

AI/ML portfolio

Resume & interviews

⚠️ Disclaimer

This application is intended only for educational and demonstration purposes.
It should not be used as a substitute for professional mental health advice.

👨‍💻 Author

Rohit Pawar
🔗 GitHub: https://github.com/rohit3576
