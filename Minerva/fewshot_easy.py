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


# --- Few-shot examples (one per language) ---
FEW_SHOT_EXAMPLES = [
    ("L'effettu di marea di i pianeti hè debuli è ùn affitteghja micca di modu significativu a forma di u Soli.", "corso"),
    ("L' àn l'è onna unitê 'd m'śura dal tèimp, pera 'l teimp ch'a gh mèt la Tera a girèr intor'n al Sōl.", "emiliano-romagnolo"),
    ("Precedentemente considerato un pianeta, da questa data Plutone fu ridefinito, assieme ad altri corpi di recente scoperta, come pianeta nano.", "italiano"),
    ("Ve songo pure ate cuorpe menore nfra ca ll'asteroide, na buona parte 'e meteoroide, 'e commete, 'e oggiette trans-nettuniane e povere spannata pe 'o spazzio.", "lombardo"),
    ("Si bene veruto â uocchio annuro Marte sbaria assaje 'e culore, se penzava pe mmiezzo d o rrovagno 'e chiante 'e staggione, ca cagnano siconn'ê periude 'e ll'anno.", "napoletano"),
    ("Is pranetas minores, is cometas, is meteoroides, is asteroides e totu su mediu interplanetariu chi b'at in mesu puru orbitant su Sole.", "piemontese"),
    ("Cô tempu si scupriu ca lu pianeta novu, chi fu chiamatu Cèriri, era lu primu corpu cilesti di na nova catigoria di corpi cilesti, assimilabbili a pianeti, ma cchiù nichi: l'astiroidi.", "siciliano"),
    ("S'Astronomia (dae su gregu: αστρονομία = astron (ἄστρον), \"istedda\" + nomos (νόμος), \"lei\"= lei de is isteddas) est sa sièntzia ch'istùdiat is acuntèssius de su celu e is ogetus celestes (isteddas, pranetas, cometas e galàssias).", "sardo"),
    ("Xe sta conosua la prexensa de n'altro pianeta nano, el so nome xe Sedna, che el xira intorno al sole a na distansa mèdia de 13 miliardi de km (3 olte pì distante de pluton).", "veneto"),
]


def build_prompt(frase):
    # Build few-shot examples as alternating user/assistant turns
    messages = []
    for esempio, lingua in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"In quale lingua è scritta la seguente frase?\nRispondi solo con questa riga, non aggiungere altro:\n\nLa frase è scritta in <lingua>.\n\nFrase: {esempio}"
        })
        messages.append({
            "role": "assistant",
            "content": f"La frase è scritta in {lingua}."
        })
    # Add the actual query
    messages.append({
        "role": "user",
        "content": f"In quale lingua è scritta la seguente frase?\nRispondi solo con questa riga, non aggiungere altro:\n\nLa frase è scritta in <lingua>.\n\nFrase: {frase}"
    })
    return messages


def parse_response(output):
    """
    HuggingFace chat pipeline returns a list of dicts like:
      [{"generated_text": [{"role": "user", ...}, {"role": "assistant", "content": "..."}]}]
    We extract the last message's content and parse it.
    """
    try:
        text = output[0]["generated_text"][-1]["content"]
    except (IndexError, KeyError, TypeError):
        return "UNKNOWN"

    match = re.search(r"([^\W\d_]+(?:-[^\W\d_]+)*)\.?$", text, re.IGNORECASE)
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

# Build all conversations upfront (now as multi-turn few-shot)
conversations = [build_prompt(frase) for frase in test_inputs]

# --- DEBUG: stampa i primi 9 output raw ---
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
    for out in outputs:
        if isinstance(out, list):
            raw_outputs.append(out)
        else:
            raw_outputs.append([out])
    print(f"Processed {min(i + BATCH_SIZE, len(conversations))} / {len(conversations)}")

# --- Parse & normalise ---
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

elapsed = time.time() - start_time
print(f"Tempo di inferenza: {elapsed:.1f}s ({elapsed/60:.1f} min)")

# --- Confusion Matrix ---
cm = confusion_matrix(test_labels_norm, predictions, labels=labels_sorted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
ax.set_title("Confusion Matrix – Minerva Few-Shot")
plt.tight_layout()
plt.savefig("confusion_matrix_minerva_fewshot.png", dpi=150)
plt.show()

# --- Save errors ---
with open("errors_minerva_fewshot.json", "w", encoding="utf-8") as f:
    json.dump(errors, f, ensure_ascii=False, indent=2)

print(f"\nErrori totali: {len(errors)} / {len(test_inputs)}")
print("Salvati in errors_minerva_fewshot.json")
