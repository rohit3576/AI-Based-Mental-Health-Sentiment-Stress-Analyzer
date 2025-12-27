import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.utils import to_categorical

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("model/stress_model.h5")

# -----------------------------
# RECREATE TEST DATA (SAME LOGIC AS TRAINING)
# -----------------------------
np.random.seed(42)

sentiment_scores = np.random.rand(1000)

stress_labels = []
for score in sentiment_scores:
    if score >= 0.6:
        stress_labels.append(0)   # Low
    elif score >= 0.4:
        stress_labels.append(1)   # Medium
    else:
        stress_labels.append(2)   # High

y_true = np.array(stress_labels)
y_true_cat = to_categorical(y_true, num_classes=3)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred_prob = model.predict(sentiment_scores)
y_pred = np.argmax(y_pred_prob, axis=1)

# -----------------------------
# METRICS
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("\n📊 Stress Model Evaluation Metrics")
print("----------------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Stress Model Confusion Matrix")
plt.show()

# -----------------------------
# CLASSIFICATION REPORT (CSV)
# -----------------------------
import pandas as pd

report = classification_report(
    y_true,
    y_pred,
    target_names=["Low", "Medium", "High"],
    output_dict=True
)

df = pd.DataFrame(report).transpose()
df.to_csv("evaluation_report_stress.csv")

print("\n📁 Stress evaluation report saved as: evaluation_report_stress.csv")
