"""
Zero-shot punctuation restoration with KyrgyzBERT (MLM).

Protocol:
  - Use the pre-trained KyrgyzBERT masked-LM checkpoint AS-IS, no fine-tuning.
  - For each test sentence, insert a [MASK] token after every word.
  - In a single forward pass, read off the softmax distribution at each mask.
  - For each mask, compare P(. | ctx), P(, | ctx), P(? | ctx) to the
    aggregate probability of any other (non-punctuation) token.
  - Predict argmax over {O, PERIOD, COMMA, QUESTION}.

Evaluated on the SAME augmented test split used for the fine-tuned models
(seed=42, 15% test).

Usage (from repo root):
    python scripts/zero_shot_kyrgyzbert.py \\
        --data data/train_data_augmented.json \\
        --out results/results_kbert_zeroshot.json
"""

import argparse
import json
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForMaskedLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='metinovadilet/KyrgyzBert')
    p.add_argument('--data',  default='data/train_data_augmented.json')
    p.add_argument('--out',   default='results_kbert_zeroshot.json')
    p.add_argument('--seed',  type=int, default=42)
    return p.parse_args()


args = parse_args()
MODEL_NAME = args.model
DATA = args.data
LABELS = ['O', 'COMMA', 'PERIOD', 'QUESTION']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── 1. Load MLM ──────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} as a masked-LM (zero-shot; no fine-tuning)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()

mask_id = tokenizer.mask_token_id
period_id = tokenizer.convert_tokens_to_ids('.')
comma_id = tokenizer.convert_tokens_to_ids(',')
question_id = tokenizer.convert_tokens_to_ids('?')
print(f"Token IDs — [MASK]={mask_id}, '.'={period_id}, ','={comma_id}, '?'={question_id}")
assert mask_id is not None and period_id is not None \
    and comma_id is not None and question_id is not None, \
    "KyrgyzBERT tokenizer missing one of [MASK], '.', ',', '?'"

# ── 2. Load & split data (identical split to fine-tuned runs) ────────
def extract(text):
    words, labels = [], []
    for token in text.split():
        word = token.rstrip('.,?!;:\'\"“”–—…()[]{}«»')
        if not word:
            continue
        trailing = token[len(word):]
        lab = 'O'
        for c in trailing:
            if c == '.': lab = 'PERIOD'; break
            elif c == ',': lab = 'COMMA'; break
            elif c == '?': lab = 'QUESTION'; break
        words.append(word); labels.append(lab)
    return words, labels

with open(DATA) as f:
    raw = json.load(f)

all_w, all_l = [], []
for e in raw:
    w, l = extract(e['text'][0])
    if len(w) >= 2:
        all_w.append(w); all_l.append(l)

_, te_w, _, te_l = train_test_split(all_w, all_l, test_size=0.15, random_state=args.seed)
print(f"Test sentences: {len(te_w)}  tokens: {sum(len(s) for s in te_w)}")
print(f"Test label distribution: {Counter(l for s in te_l for l in s)}")

# ── 3. Zero-shot prediction ──────────────────────────────────────────
MAX_LEN = 512  # KyrgyzBERT supports 512

def predict_sentence(words):
    """Return list of predicted labels, one per word."""
    # Build "w1 [MASK] w2 [MASK] ... wn [MASK]"
    # so each gap (incl. after last word) has a mask predicting what follows.
    interleaved = []
    for w in words:
        interleaved.append(w)
        interleaved.append(tokenizer.mask_token)

    # Tokenize
    enc = tokenizer(interleaved, is_split_into_words=True,
                    return_tensors='pt', truncation=True, max_length=MAX_LEN)
    input_ids = enc['input_ids'].to(device)

    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]  # [seq, vocab]

    # Find mask positions and which word they correspond to
    word_ids = enc.word_ids(batch_index=0)
    mask_positions = []       # token-level positions of [MASK]
    mask_word_idx = []        # corresponding index in `words` list
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        # odd indices in `interleaved` are the MASK words (1, 3, 5, ...)
        if wid % 2 == 1:
            mask_positions.append(tok_idx)
            mask_word_idx.append(wid // 2)

    preds = ['O'] * len(words)
    for tok_idx, wid in zip(mask_positions, mask_word_idx):
        if wid >= len(preds):
            continue
        probs = torch.softmax(logits[tok_idx], dim=-1)
        p_period   = probs[period_id].item()
        p_comma    = probs[comma_id].item()
        p_question = probs[question_id].item()
        p_other    = 1.0 - (p_period + p_comma + p_question)

        scores = {
            'O':        p_other,
            'PERIOD':   p_period,
            'COMMA':    p_comma,
            'QUESTION': p_question,
        }
        preds[wid] = max(scores, key=scores.get)
    return preds

# ── 4. Run over test set ─────────────────────────────────────────────
all_true, all_pred = [], []
for i, (words, labels) in enumerate(zip(te_w, te_l)):
    preds = predict_sentence(words)
    all_true.extend(labels)
    all_pred.extend(preds)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(te_w)}")

# ── 5. Report ─────────────────────────────────────────────────────────
report = classification_report(all_true, all_pred, labels=LABELS,
                                digits=3, output_dict=True, zero_division=0)
print("\n" + "=" * 60)
print("KyrgyzBERT ZERO-SHOT (no fine-tuning)")
print("=" * 60)
print(classification_report(all_true, all_pred, labels=LABELS,
                             digits=3, zero_division=0))

cm = confusion_matrix(all_true, all_pred, labels=LABELS)
print("Confusion matrix (rows=true, cols=pred), labels =", LABELS)
print(cm)

out = {
    'model': MODEL_NAME,
    'protocol': 'zero-shot MLM mask-fill',
    'n_test_tokens': len(all_true),
    'weighted': {
        'precision': round(report['weighted avg']['precision'], 4),
        'recall':    round(report['weighted avg']['recall'], 4),
        'f1':        round(report['weighted avg']['f1-score'], 4),
    },
    'per_class': {
        c: {
            'precision': round(report[c]['precision'], 4),
            'recall':    round(report[c]['recall'], 4),
            'f1':        round(report[c]['f1-score'], 4),
            'support':   int(report[c]['support']),
        } for c in LABELS
    },
    'confusion_matrix_raw': cm.tolist(),
    'confusion_matrix_labels': LABELS,
}
with open(args.out, 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {args.out}")
print(f"\nWeighted F1 = {out['weighted']['f1']:.3f}")
print(f"QUESTION F1 = {out['per_class']['QUESTION']['f1']:.3f}")
