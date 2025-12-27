import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -----------------------------
# CONFIG (MUST MATCH TRAINING)
# -----------------------------
VOCAB_SIZE = 10000
MAX_LEN = 200

# -----------------------------
# LOAD TEST DATA
# -----------------------------
(_, _), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

x_test = pad_sequences(
    x_test,
    maxlen=MAX_LEN,
    padding="post"
)

print("Test data shape:", x_test.shape)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("model/sentiment_model.h5")

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# -----------------------------
# METRICS
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Sentiment Model Evaluation Metrics")
print("----------------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Sentiment Model Confusion Matrix")
plt.show()
from sklearn.metrics import roc_curve, auc

# -----------------------------
# ROC CURVE & AUC
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Sentiment Model")
plt.legend(loc="lower right")
plt.show()

print(f"\nROC-AUC Score: {roc_auc:.4f}")

from sklearn.metrics import classification_report
import pandas as pd

# -----------------------------
# CLASSIFICATION REPORT (CSV)
# -----------------------------
report = classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"],
    output_dict=True
)

df_report = pd.DataFrame(report).transpose()
df_report.to_csv("evaluation_report_sentiment.csv")

print("\n📁 Evaluation report saved as: evaluation_report_sentiment.csv")
