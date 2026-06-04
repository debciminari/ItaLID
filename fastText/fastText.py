import fasttext
import numpy as np
import json
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from itertools import product

# ── Preprocessing ─────────────────────────────────────────────────────────────

def load_json(json_path):
    """Restituisce (sentences, labels) come liste parallele."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences, labels = [], []
    for obj in data:
        for sent in obj["sentences"]:
            sent = sent.lower()
            sent = " ".join(sent.split())
            if sent:
                sentences.append(sent)
                labels.append(obj["lang"])
    return sentences, labels


def write_fasttext_file(sentences, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        for sent, label in zip(sentences, labels):
            f.write(f"__label__{label} {sent}\n")


# ── Caricamento dati ──────────────────────────────────────────────────────────

train_sentences, train_labels = load_json("train.json")
test_sentences,  test_labels  = load_json("test.json")

# ── 5-Fold CV + Grid Search ───────────────────────────────────────────────────

lr_values    = [0.1, 0.5, 1.0]
epoch_values = [10, 25, 50]
param_grid   = list(product(lr_values, epoch_values))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("── 5-Fold CV Grid Search ──")
print(f"{'lr':>6} {'epochs':>7}  {'F1 mean':>9} {'F1 std':>8}")
print("-" * 38)

best_f1_mean = 0
best_params  = {}
cv_results   = []

for lr, epochs in param_grid:
    fold_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(train_sentences, train_labels)
    ):
        fold_train_sents  = [train_sentences[i] for i in train_idx]
        fold_train_labels = [train_labels[i]    for i in train_idx]
        fold_val_sents    = [train_sentences[i] for i in val_idx]
        fold_val_labels   = [train_labels[i]    for i in val_idx]

        # File temporanei per il fold
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
        write_fasttext_file(fold_train_sents, fold_train_labels, tmp_path)

        model = fasttext.train_supervised(
            input=tmp_path,
            lr=lr,
            epoch=epochs,
            verbose=0,
        )
        os.unlink(tmp_path)

        preds = [
            model.predict(s)[0][0].replace("__label__", "")
            for s in fold_val_sents
        ]
        fold_f1s.append(f1_score(fold_val_labels, preds, average="macro"))

    mean_f1 = np.mean(fold_f1s)
    std_f1  = np.std(fold_f1s)
    cv_results.append({"lr": lr, "epoch": epochs, "mean_f1": mean_f1, "std_f1": std_f1})
    print(f"{lr:>6.1f} {epochs:>7}  {mean_f1:>9.4f} {std_f1:>8.4f}")

    if mean_f1 > best_f1_mean:
        best_f1_mean = mean_f1
        best_params  = {"lr": lr, "epoch": epochs}

print(f"\nMigliori parametri (CV): {best_params}  →  F1 mean={best_f1_mean:.4f}")

# ── Valutazione finale sul test set (N run per mean/std) ─────────────────────

N_RUNS = 5
all_acc, all_prec, all_rec, all_f1 = [], [], [], []
all_predictions = []

# Scriviamo il training completo su file
train_txt = "train_full.txt"
write_fasttext_file(train_sentences, train_labels, train_txt)

print(f"\n── Test-set evaluation: {N_RUNS} runs con best params ──")

for run in range(N_RUNS):
    model = fasttext.train_supervised(
        input=train_txt,
        seed=run,
        verbose=0,
        **best_params,
    )
    preds = [
        model.predict(s)[0][0].replace("__label__", "")
        for s in test_sentences
    ]
    all_predictions.append(preds)
    all_acc.append(accuracy_score(test_labels, preds))
    all_prec.append(precision_score(test_labels, preds, average="macro"))
    all_rec.append(recall_score(test_labels, preds, average="macro"))
    all_f1.append(f1_score(test_labels, preds, average="macro"))

print(f"Accuracy:  {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"Precision: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
print(f"Recall:    {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
print(f"F1:        {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

# Run con F1 più vicina alla media → usata per report e confusion matrix
best_run_idx = int(np.argmin(np.abs(np.array(all_f1) - np.mean(all_f1))))
best_preds   = all_predictions[best_run_idx]
print(f"\n(Report e confusion matrix dalla run {best_run_idx} "
      f"— F1={all_f1[best_run_idx]:.4f}, la più vicina alla media)")


# ── Per-class report con mean ± std per classe ───────────────────────────────

from collections import defaultdict

# Raccogliamo precision/recall/f1 per classe su tutte le run
class_metrics = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})
labels_order = sorted(set(test_labels))

for preds in all_predictions:
    report = classification_report(
        test_labels, preds,
        labels=labels_order,
        output_dict=True,
        zero_division=0,
    )
    for label in labels_order:
        class_metrics[label]["precision"].append(report[label]["precision"])
        class_metrics[label]["recall"].append(report[label]["recall"])
        class_metrics[label]["f1"].append(report[label]["f1-score"])

print("\n── Per-class report (mean ± std across runs) ──")
header = f"{'label':<12} {'precision':>18} {'recall':>18} {'f1-score':>18}"
print(header)
print("-" * len(header))

for label in labels_order:
    m = class_metrics[label]
    p  = f"{np.mean(m['precision']):.4f} ± {np.std(m['precision']):.4f}"
    r  = f"{np.mean(m['recall']):.4f} ± {np.std(m['recall']):.4f}"
    f  = f"{np.mean(m['f1']):.4f} ± {np.std(m['f1']):.4f}"
    print(f"{label:<12} {p:>18} {r:>18} {f:>18}")


# ── Predizioni ed errori ────────────────────────────────────────────────────────────────────

preds = []

for sent, label, pred in zip(test_inputs, test_labels, predictions):
    preds.append({
            "sentence": sent,
            "true": label,
            "predicted": pred
            })

PREDS_OUT = "predictions_fasttext.json"
with open(PREDS_OUT, "w", encoding="utf-8") as f:
    json.dump(preds, f, ensure_ascii=False, indent=2)

print(f"Predizioni: {len(preds)}")


errors = [
    {"sentence": sent, "true": true, "predicted": pred}
    for sent, true, pred in zip(test_sentences, test_labels, best_preds)
    if true != pred
]

ERRORS_OUT = "errors_fasttext.json"
with open(ERRORS_OUT, "w", encoding="utf-8") as f:
    json.dump(errors, f, ensure_ascii=False, indent=2)

print(f"\nErrori: {len(errors)} / {len(test_sentences)} "
      f"({len(errors) / len(test_sentences) * 100:.1f}%)")
print(f"Salvati in: {ERRORS_OUT}")

# ── Confusion matrix ──────────────────────────────────────────────────────────

labels_order = sorted(set(test_labels))
cm = confusion_matrix(test_labels, best_preds, labels=labels_order)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_order,
    yticklabels=labels_order,
    ax=ax,
)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title("Confusion Matrix – fastText", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix_fasttext.png", dpi=150)
plt.show()
