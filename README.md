🚀 Live Demo

🌐 Live App:
👉 https://ai-based-mental-health-sentiment-stress.onrender.com

https://<your-app-name>.onrender.com

📌 Project Overview

Mental health awareness is critical in today’s fast-paced world. This project uses Natural Language Processing (NLP) and Deep Learning to analyze a user’s thoughts and emotions expressed through text.

The system:

Detects sentiment (Positive / Negative)

Estimates stress level (Low / Medium / High)

Displays confidence score

Provides wellness suggestions

Features a modern glassmorphism UI

⚠️ This tool is for educational purposes only and is not a medical diagnosis.

✨ Key Features

🧠 AI-powered sentiment analysis using Bidirectional LSTM

📊 Stress level classification (Low / Medium / High)

🎯 Confidence percentage visualization

🎨 Modern glassmorphism UI (responsive for mobile & desktop)

⏳ Loading animation for better UX

☁️ Free cloud deployment (Render)

🛠️ Tech Stack
🔹 Backend

Python

Flask

TensorFlow / Keras

🔹 Machine Learning

Bidirectional LSTM (BiLSTM)

IMDB Dataset (for sentiment learning)

Text tokenization & padding

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
Tokenization & Padding
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

📂 Project Structure
AI-Based-Mental-Health-Sentiment-Stress-Analyzer/
│
├── app.py
├── requirements.txt
│
├── model/
│   ├── sentiment_model.h5
│   ├── stress_model.h5
│   └── preprocess_config.pkl
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
source venv/bin/activate   # Windows: venv\Scripts\activate

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

No paid services required.

🎓 Academic & Interview Relevance

This project demonstrates:

NLP preprocessing

Deep learning with LSTM

Model evaluation & selection

Flask backend integration

UI/UX design

Free cloud deployment

Perfect for:

College final-year project

AI/ML portfolio

Resume & interviews

⚠️ Disclaimer

This application is intended only for educational and demonstration purposes.
It should not be used as a substitute for professional mental health advice.

👨‍💻 Author

Rohit Pawar
GitHub: https://github.com/rohit3576

⭐ If you like this project

Give it a ⭐ on GitHub
