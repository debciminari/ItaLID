## Multinomial Naive Bayes — char 5-gram + word bigram (2,2)
## FeatureUnion: CountVectorizer char (5,5) + CountVectorizer word (2,2)

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

MODEL_NAME = "char5_word_bigram"
OUTPUT_DIR = "./5_splits2/esperimento2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD TRAIN
# ==========================================

with open("5_splits2/train.json", "r", encoding="utf-8") as t:
    train_set = json.load(t)

inputs = []
labels = []

for obj in train_set:
    for sent in obj["sentences"]:
        inputs.append(sent.lower())
        labels.append(obj["lang"])

print(inputs[:2])
print(labels[:2])
print(f"Length of inputs: {len(inputs)}")
print(f"Length of labels: {len(labels)}")

# ==========================================
# LOAD TEST
# ==========================================

with open("5_splits2/test.json", encoding="utf-8") as f:
    test_data = json.load(f)

test_inputs = []
test_labels = []

for obj in test_data:
    for sent in obj["sentences"]:
        test_inputs.append(sent.lower())
        test_labels.append(obj["lang"])

# ==========================================
# PIPELINE con FeatureUnion
# ==========================================

feature_union = FeatureUnion([
    (
        "char5",
        CountVectorizer(
            analyzer="char",
            ngram_range=(5, 5)
        )
    ),
    (
        "word_bi",
        CountVectorizer(
            analyzer="word",
            ngram_range=(2, 2)
        )
    )
])

pipeline = Pipeline([
    ("features", feature_union),
    ("nb", MultinomialNB())
])

# ==========================================
# GRID SEARCH (solo alpha)
# ==========================================

param_grid = {
    "nb__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print(f"\n{'='*50}")
print(f"  Modello: char 5-gram + word bigram")
print(f"{'='*50}")

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    verbose=2,
    n_jobs=-1
)

grid.fit(inputs, labels)

print("\nBest params:")
print(grid.best_params_)

print("\nBest CV score:")
print(grid.best_score_)

# ==========================================
# RESULTS (MEAN and STDEV)
# ==========================================

results = pd.DataFrame(grid.cv_results_)
summary_alpha = results[["param_nb__alpha", "mean_test_score", "std_test_score"]]
print(summary_alpha.to_string(index=False))

# ==========================================
# BEST MODEL — PREDICTION
# ==========================================

best_model = grid.best_estimator_
predictions = best_model.predict(test_inputs)

# ==========================================
# SAVE ERRORS
# ==========================================

errors = []

for sent, true_lang, pred_lang in zip(test_inputs, test_labels, predictions):
    if true_lang != pred_lang:
        errors.append({
            "sentence": sent,
            "true": true_lang,
            "predicted": pred_lang
        })

errors_path = os.path.join(OUTPUT_DIR, f"errors_{MODEL_NAME}.json")
with open(errors_path, "w", encoding="utf-8") as f:
    json.dump(errors, f, ensure_ascii=False, indent=2)

# ==========================================
# RESULTS
# ==========================================

print(
    f"\nErrori: {len(errors)} / {len(test_inputs)} "
    f"({len(errors)/len(test_inputs)*100:.2f}%)"
)

print("\nClassification Report:\n")
print(classification_report(test_labels, predictions))

report = classification_report(test_labels, predictions, output_dict=True)

summary = {
    "model":      MODEL_NAME,
    "best_alpha": grid.best_params_["nb__alpha"],
    "cv_f1":      round(grid.best_score_, 4),
    "test_f1":    round(report["macro avg"]["f1-score"], 4),
    "test_acc":   round(report["accuracy"], 4),
    "errors":     len(errors),
}

print("\nRiepilogo:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# ==========================================
# CONFUSION MATRIX
# ==========================================

labels_order = sorted(set(test_labels))
cm = confusion_matrix(test_labels, predictions, labels=labels_order)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels_order,
    yticklabels=labels_order,
    ax=ax
)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(
    f"Normalized Confusion Matrix — {MODEL_NAME}",
    fontsize=14
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{MODEL_NAME}.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nConfusion matrix salvata in: {cm_path}")