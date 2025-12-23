import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# -----------------------------
# APP INIT
# -----------------------------
app = Flask(__name__)

# -----------------------------
# LOAD MODELS
# -----------------------------
sentiment_model = tf.keras.models.load_model("model/sentiment_model.h5")
stress_model = tf.keras.models.load_model(
    "model/stress_model.h5"
)  # kept for architecture completeness

# Load preprocessing config
with open("model/preprocess_config.pkl", "rb") as f:
    config = pickle.load(f)

VOCAB_SIZE = config["vocab_size"]
MAX_LEN = config["max_len"]

# IMDB word index
word_index = imdb.get_word_index()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def encode_text(text):
    words = text.lower().split()
    encoded = []

    for word in words:
        idx = word_index.get(word, 2)  # 2 = <UNK>
        if idx < VOCAB_SIZE:
            encoded.append(idx)
        else:
            encoded.append(2)

    # Handle empty or very short input
    if len(encoded) == 0:
        encoded = [2]

    padded = pad_sequences(
        [encoded],
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )
    return padded


def get_stress_label(score):
    if score >= 0.6:
        return "Low Stress 😊", "You're doing well. Keep maintaining balance."
    elif score >= 0.4:
        return "Medium Stress 😐", "Try relaxation, short breaks, and deep breathing."
    else:
        return "High Stress 😟", "Consider rest, talking to someone, or mindfulness."

# -----------------------------
# ROUTE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    stress = None
    advice = None
    confidence = None

    if request.method == "POST":
        user_text = request.form["text"]

        encoded_text = encode_text(user_text)
        sentiment_score = float(sentiment_model.predict(encoded_text)[0][0])

        # Confidence percentage
        confidence = round(sentiment_score * 100, 2)

        # Sentiment label
        sentiment = "Positive 😊" if sentiment_score >= 0.5 else "Negative 😔"

        # Stress level + advice
        stress, advice = get_stress_label(sentiment_score)

    return render_template(
        "index.html",
        sentiment=sentiment,
        stress=stress,
        advice=advice,
        confidence=confidence
    )

# -----------------------------
# RUN (RENDER + LOCAL SAFE)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
