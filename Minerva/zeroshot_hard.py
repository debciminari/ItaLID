import transformers
import torch
import json
import re
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import time

# --- GPU info ---
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("Nessuna GPU disponibile, uso CPU")


ISO_TO_LANG = {
    "it": "italiano",
    "eml": "emiliano-romagnolo",
    "pms": "piemontese",
    "co": "corso",
    "nap": "napoletano",
    "sc": "sardo",
    "scn": "siciliano",
    "lmo": "lombardo",
    "vec": "veneto"
}

with open("test.json", "r", encoding="utf-8") as f:
    test_set = json.load(f)

test_inputs = []
test_labels = []

for obj in test_set:
    label = obj["lang"]
    for sent in obj["sentences"]:
        test_inputs.append(sent)
        test_labels.append(label)

test_labels = [ISO_TO_LANG[code] for code in test_labels]


def build_prompt(frase):
    return f"""Agisci come un sistema esperto di linguistica dei dialetti italiani. In quale lingua è scritta la seguente frase?
    Rispondi solo con questa riga, non aggiungere altro:

    La frase è scritta in <lingua>.

    Frase: {frase}"""


def parse_response(output):
    """
    HuggingFace chat pipeline returns a list of dicts like:
      [{"generated_text": [{"role": "user", ...}, {"role": "assistant", "content": "..."}]}]
    We extract the last message's content and parse it.
    """
    try:
        # output is a list (one item per input in the batch)
        text = output[0]["generated_text"][-1]["content"]
    except (IndexError, KeyError, TypeError):
        return "UNKNOWN"

    match = re.search(r"([^\W\d_]+)\.?$", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return "UNKNOWN"


# --- Model ---
model_id = "sapienzanlp/Minerva-7B-instruct-v1.0"

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    do_sample=False,
    temperature=None,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
    )

pipe.tokenizer.padding_side = "left"

print("MODELLO CARICATO")

# --- Batched inference ---
BATCH_SIZE = 32

# Build all conversations upfront
conversations = [
    [{"role": "user", "content": build_prompt(frase)}]
    for frase in test_inputs
]

# --- DEBUG: stampa i primi 5 output raw ---
debug_batch = conversations[:9]
debug_out = pipe(debug_batch, max_new_tokens=20, batch_size=9)

for i, out in enumerate(debug_out):
    if isinstance(out, list):
        content = out[0]["generated_text"][-1]["content"]
    else:
        content = out["generated_text"][-1]["content"]
    print(f"\n--- Esempio {i+1} ---")
    print(f"Input: {test_inputs[i]}")
    print(f"Output raw:\n{content}")
    print(f"Parsed: {parse_response(out if isinstance(out, list) else [out])}")

# Avvia timer prima del loop
start_time = time.time()

raw_outputs = []
for i in range(0, len(conversations), BATCH_SIZE):
    batch = conversations[i : i + BATCH_SIZE]
    outputs = pipe(batch, max_new_tokens=20)
    # pipe returns a list-of-lists when given a list of conversations;
    # flatten one level so each element corresponds to one conversation.
    for out in outputs:
        # out may be a list (one result per conversation item) or a dict
        if isinstance(out, list):
            raw_outputs.append(out)
        else:
            raw_outputs.append([out])
    print(f"Processed {min(i + BATCH_SIZE, len(conversations))} / {len(conversations)}")

# --- Parse & normalise ---
# Normalise labels to lowercase for consistent comparison
test_labels_norm = [l.lower() for l in test_labels]

predictions = []
errors = []

for frase, label, out in zip(test_inputs, test_labels_norm, raw_outputs):
    prediction = parse_response(out)
    predictions.append(prediction)

    if prediction != label:
        errors.append({
            "frase": frase,
            "true_label": label,
            "predicted_label": prediction,
            "full_response": out
        })

# --- Metrics ---
labels_sorted = sorted(set(test_labels_norm))

f1        = f1_score(test_labels_norm, predictions, labels=labels_sorted, average="weighted", zero_division=0)
precision = precision_score(test_labels_norm, predictions, labels=labels_sorted, average="weighted", zero_division=0)
recall    = recall_score(test_labels_norm, predictions, labels=labels_sorted, average="weighted", zero_division=0)


from sklearn.metrics import classification_report

# --- Per-class metrics ---
report = classification_report(
    test_labels_norm,
    predictions,
    labels=labels_sorted,
    zero_division=0
)
print("\nPer-class metrics:")
print(report)


print(f"\nF1 (weighted):        {f1:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted):    {recall:.4f}")

# --- Tempo totale ---
elapsed = time.time() - start_time
print(f"Tempo di inferenza: {elapsed:.1f}s ({elapsed/60:.1f} min)")

# --- Confusion Matrix ---
cm = confusion_matrix(test_labels_norm, predictions, labels=labels_sorted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_minerva_zero_1_diff.png", dpi=150)
plt.show()

# --- Save errors ---
with open("errors_minerva_zero_1_diff.json", "w", encoding="utf-8") as f:
    json.dump(errors, f, ensure_ascii=False, indent=2)

print(f"\nErrori totali: {len(errors)} / {len(test_inputs)}")
print("Salvati in errors_minerva_zero.json")

