## Multinomial Naive Bayes con tuning alpha

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd
import json


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
# PIPELINE
# ==========================================

pipeline = Pipeline([
    (
        "vect",
        CountVectorizer(
            token_pattern=r"[a-zA-ZÀ-ÿ]+",
            analyzer='word',
            ngram_range=(1, 2)
        )
    ),
    (
        "nb",
        MultinomialNB()
    )
])


# ==========================================
# GRID SEARCH
# ==========================================

param_grid = {
    "nb__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    verbose=2,
    n_jobs=-1
)


# ==========================================
# TRAINING + TUNING
# ==========================================

grid.fit(inputs, labels)

print("\nBest params:")
print(grid.best_params_)

print("\nBest CV score:")
print(grid.best_score_)

# ==========================================
# RESULTS (MEAN and STDEV) (!!! calcolati sul fold che funge da validation)
# ==========================================

results = pd.DataFrame(grid.cv_results_)
summary = results[['param_nb__alpha', 'mean_test_score', 'std_test_score']]
print(summary.to_string(index=False))

# ==========================================
# BEST MODEL
# ==========================================

best_model = grid.best_estimator_

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
# PREDICTION
# ==========================================

predictions = best_model.predict(test_inputs)


# ==========================================
# SAVE ERRORS
# ==========================================

errors = []

for sent, true_lang, pred_lang in zip(
    test_inputs,
    test_labels,
    predictions
):
    if true_lang != pred_lang:
        errors.append({
            "sentence": sent,
            "true": true_lang,
            "predicted": pred_lang
        })

with open(
    "5_splits2/esperimento2/errors_UNI+BI.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(errors, f, ensure_ascii=False, indent=2)


# ==========================================
# RESULTS
# ==========================================

print(
    f"Errori: {len(errors)} / {len(test_inputs)} "
    f"({len(errors)/len(test_inputs)*100:.2f}%)"
)

print("\nClassification Report:\n")

print(classification_report(
    test_labels,
    predictions
))



# ==========================================
# CONFUSION MATRIX
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ordine consistente delle classi
labels_order = sorted(set(test_labels))

# confusion matrix
cm = confusion_matrix(
    test_labels,
    predictions,
    labels=labels_order
)

# ==========================================
# NORMALIZZAZIONE
# ==========================================

cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

# ==========================================
# PLOT
# ==========================================

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
    "Normalized Confusion Matrix",
    fontsize=14
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_uni+bi.png",
    dpi=150
)

plt.show()