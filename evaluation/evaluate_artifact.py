"""
Re-evaluate the shipped BiGRU artifact on the dair-ai/emotion test split.

Writes static/data/evaluation.json, which the frontend's Evaluation section reads.
It does not train or modify anything. Two input paths are scored:

  * "api"      – text cleaned with main.preprocess_text (exactly what POST /predict does)
  * "notebook" – raw dataset text, as in final_clean.ipynb (no extra cleaning)

Run from the project root:
    pip install -r evaluation/requirements-eval.txt
    python evaluation/evaluate_artifact.py
"""
import json
import os
import pickle
import sys
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from keras.models import load_model  # noqa: E402
from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: E402

from main import emotion_labels, max_sequence_length, model_path, preprocess_text, tokenizer_path  # noqa: E402


def metrics(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"label": emotion_labels[c], "precision": prec, "recall": rec, "f1": f1, "support": int(cm[c].sum())})
    supports = np.array([p["support"] for p in per_class])
    f1s = np.array([p["f1"] for p in per_class])
    return {
        "accuracy": float((np.array(y_true) == np.array(y_pred)).mean()),
        "macro_f1": float(f1s.mean()),
        "weighted_f1": float((f1s * supports).sum() / supports.sum()),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def predict(model, tokenizer, texts):
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_sequence_length, padding="post", truncating="post")
    return model.predict(padded, batch_size=256, verbose=0)


def main():
    ds = load_dataset("dair-ai/emotion", split="test")
    names = ds.features["label"].names
    assert names == emotion_labels, f"label order mismatch: {names} vs {emotion_labels}"
    texts, labels = list(ds["text"]), [int(x) for x in ds["label"]]

    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    probs_api = predict(model, tokenizer, [preprocess_text(t) for t in texts])
    probs_raw = predict(model, tokenizer, texts)
    pred_api = probs_api.argmax(1)
    pred_raw = probs_raw.argmax(1)
    k = len(emotion_labels)

    conf = probs_api.max(1)
    correct = pred_api == np.array(labels)
    bins = np.linspace(0, 1, 11)
    reliability = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            reliability.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                                "mean_confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    ece = sum(r["n"] * abs(r["accuracy"] - r["mean_confidence"]) for r in reliability) / len(labels)

    errors = []
    order = np.argsort(-conf)
    for i in order:
        if not correct[i]:
            errors.append({"text": texts[i], "true": emotion_labels[labels[i]], "predicted": emotion_labels[pred_api[i]],
                           "confidence": float(conf[i])})
        if len(errors) >= 12:
            break

    vocab_size = tokenizer.num_words
    oov_rates = []
    for t in texts:
        words = preprocess_text(t).split()
        if words:
            idx = [tokenizer.word_index.get(w) for w in words]
            oov_rates.append(sum(1 for i in idx if i is None or i >= vocab_size) / len(words))

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "evaluation/evaluate_artifact.py",
        "artifact": os.path.basename(model_path),
        "dataset": "dair-ai/emotion (split config), test split",
        "n": len(labels),
        "labels": emotion_labels,
        "class_counts": np.bincount(labels, minlength=k).tolist(),
        "api_path": metrics(labels, pred_api, k),
        "notebook_path": {"accuracy": float((pred_raw == np.array(labels)).mean()),
                          "macro_f1": metrics(labels, pred_raw, k)["macro_f1"]},
        "agreement_api_vs_notebook": float((pred_api == pred_raw).mean()),
        "calibration": {"bins": reliability, "ece": float(ece)},
        "high_confidence_errors": errors,
        "tokenizer": {"num_words": vocab_size, "word_index_size": len(tokenizer.word_index),
                      "mean_oov_rate_test": float(np.mean(oov_rates))},
        "model": {"trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                  "max_sequence_length": max_sequence_length},
    }
    os.makedirs(os.path.join(ROOT, "static", "data"), exist_ok=True)
    path = os.path.join(ROOT, "static", "data", "evaluation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"API path: acc={out['api_path']['accuracy']:.4f} macroF1={out['api_path']['macro_f1']:.4f}")
    print(f"Notebook path: acc={out['notebook_path']['accuracy']:.4f}; agreement={out['agreement_api_vs_notebook']:.4f}; ECE={ece:.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
