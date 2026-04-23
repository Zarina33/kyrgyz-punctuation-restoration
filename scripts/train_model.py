"""
Unified fine-tuning script for Kyrgyz punctuation restoration. Used for all
three transformer models (mBERT, XLM-RoBERTa, KyrgyzBERT) and both dataset
versions (original 14,028 sentences, augmented 16,028 sentences) to reproduce
Tables 4 and 5 of the paper.

Usage (from repo root):
    # Augmented-data runs (main results)
    python scripts/train_model.py --model bert-base-multilingual-cased \\
        --data data/train_data_augmented.json --tag mbert_aug
    python scripts/train_model.py --model xlm-roberta-base \\
        --data data/train_data_augmented.json --tag xlmr_aug
    python scripts/train_model.py --model metinovadilet/KyrgyzBert \\
        --data data/train_data_augmented.json --tag kbert_aug

    # Original-data runs (for the ablation in Table 10)
    python scripts/train_model.py --model bert-base-multilingual-cased \\
        --data data/train_data_original.json --tag mbert_orig
    # ... etc.

Identical setup across models for a fair comparison:
- Seed 42, split 76.5 / 8.5 / 15, max_len 256, batch 16, lr 5e-5, 5 epochs, FP16.
- Outputs: ./ckpt_<tag>/, ./confusion_matrix_<tag>.png, ./results_<tag>.json.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   help='HF model name (e.g. xlm-roberta-base, bert-base-multilingual-cased, metinovadilet/KyrgyzBert)')
    p.add_argument('--data', required=True,
                   help='Path to training json (train_data.json or train_data_augmented.json)')
    p.add_argument('--tag', required=True,
                   help='Short tag for output dir & result file (e.g. xlmr_orig, mbert_aug)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


LABEL2ID = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def extract_tokens_and_labels(text):
    raw_tokens = text.split()
    words, labels = [], []
    for token in raw_tokens:
        word = token.rstrip('.,?!;:\'\"“”–—…()[]{}«»')
        if not word:
            continue
        trailing = token[len(word):]
        label = 'O'
        for char in trailing:
            if char == '.':
                label = 'PERIOD'; break
            elif char == ',':
                label = 'COMMA'; break
            elif char == '?':
                label = 'QUESTION'; break
        words.append(word)
        labels.append(label)
    return words, labels


class PunctDataset(Dataset):
    def __init__(self, words_list, labels_list, tokenizer, max_len):
        self.words_list = words_list
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.words_list)

    def __getitem__(self, idx):
        words = self.words_list[idx]
        labels = self.labels_list[idx]
        encoding = self.tokenizer(
            words, is_split_into_words=True, truncation=True,
            max_length=self.max_len, padding='max_length', return_tensors='pt',
        )
        word_ids = encoding.word_ids(batch_index=0)
        # Label to last subtoken of each word
        label_ids = [-100] * len(word_ids)
        for i in range(len(word_ids) - 1, -1, -1):
            wid = word_ids[i]
            if wid is None:
                continue
            if i == len(word_ids) - 1 or word_ids[i + 1] != wid:
                label_ids[i] = LABEL2ID[labels[wid]]
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long),
        }


def compute_metrics_factory():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_labels, pred_labels = [], []
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i][j] != -100:
                    true_labels.append(ID2LABEL[labels[i][j]])
                    pred_labels.append(ID2LABEL[preds[i][j]])
        report = classification_report(
            true_labels, pred_labels,
            labels=['O', 'COMMA', 'PERIOD', 'QUESTION'],
            output_dict=True, zero_division=0,
        )
        return {
            'f1_weighted': report['weighted avg']['f1-score'],
            'precision_weighted': report['weighted avg']['precision'],
            'recall_weighted': report['weighted avg']['recall'],
        }
    return compute_metrics


def main():
    args = parse_args()
    print(f"\n{'='*70}\nRun tag: {args.tag}  |  Model: {args.model}  |  Data: {args.data}\n{'='*70}")

    # ── Load & process data ────────────────────────────────
    with open(args.data, 'r') as f:
        raw_data = json.load(f)
    print(f"Total sentences: {len(raw_data)}")

    all_words, all_labels, skipped = [], [], 0
    for entry in raw_data:
        text = entry['text'][0]
        words, labels = extract_tokens_and_labels(text)
        if len(words) < 2:
            skipped += 1; continue
        all_words.append(words); all_labels.append(labels)

    print(f"Processed: {len(all_words)} sentences, skipped: {skipped}")
    flat = [l for ll in all_labels for l in ll]
    print(f"Label distribution: {Counter(flat)}")

    # ── Split (seed=42, identical across all runs) ─────────
    train_w, test_w, train_l, test_l = train_test_split(
        all_words, all_labels, test_size=0.15, random_state=args.seed)
    train_w, val_w, train_l, val_l = train_test_split(
        train_w, train_l, test_size=0.1, random_state=args.seed)
    print(f"Train: {len(train_w)}, Val: {len(val_w)}, Test: {len(test_w)}")

    # ── Tokenizer / datasets ────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = PunctDataset(train_w, train_l, tokenizer, args.max_len)
    val_ds = PunctDataset(val_w, val_l, tokenizer, args.max_len)
    test_ds = PunctDataset(test_w, test_l, tokenizer, args.max_len)

    # ── Model ───────────────────────────────────────────────
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.1f}M")

    output_dir = f'./ckpt_{args.tag}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_weighted',
        greater_is_better=True,
        fp16=True,
        logging_steps=100,
        report_to='none',
        seed=args.seed,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics_factory(),
    )

    trainer.train()

    # ── Test evaluation ─────────────────────────────────────
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_true, all_pred = [], []
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu()
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i][j] != -100:
                        all_true.append(ID2LABEL[labels[i][j].item()])
                        all_pred.append(ID2LABEL[preds[i][j].item()])

    print(f"\nTotal test tokens: {len(all_true)}")
    report = classification_report(
        all_true, all_pred,
        labels=['O', 'COMMA', 'PERIOD', 'QUESTION'],
        digits=3, output_dict=True, zero_division=0,
    )
    print(classification_report(
        all_true, all_pred,
        labels=['O', 'COMMA', 'PERIOD', 'QUESTION'],
        digits=3, zero_division=0,
    ))

    # ── Confusion matrix ─────────────────────────────────────
    cm_labels = ['O', 'COMMA', 'PERIOD', 'QUESTION']
    cm_norm = confusion_matrix(all_true, all_pred, labels=cm_labels, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=cm_labels)
    disp.plot(ax=ax, cmap='Oranges', values_format='.3f')
    ax.set_title(f'{args.tag} — Confusion Matrix (Normalized)', fontsize=13)
    plt.tight_layout()
    cm_path = f'confusion_matrix_{args.tag}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")

    cm_raw = confusion_matrix(all_true, all_pred, labels=cm_labels)

    # ── Save JSON results ────────────────────────────────────
    result = {
        'tag': args.tag,
        'model': args.model,
        'data': args.data,
        'params_millions': round(n_params, 2),
        'weighted': {
            'precision': round(report['weighted avg']['precision'], 4),
            'recall': round(report['weighted avg']['recall'], 4),
            'f1': round(report['weighted avg']['f1-score'], 4),
        },
        'per_class': {
            cls: {
                'precision': round(report[cls]['precision'], 4),
                'recall': round(report[cls]['recall'], 4),
                'f1': round(report[cls]['f1-score'], 4),
                'support': int(report[cls]['support']),
            } for cls in cm_labels
        },
        'confusion_matrix_raw': cm_raw.tolist(),
        'confusion_matrix_labels': cm_labels,
    }
    result_path = f'results_{args.tag}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {result_path}")

    print("\n" + "=" * 60)
    print(f"FINAL: {args.tag}  |  F1={result['weighted']['f1']:.3f}  |  QUESTION F1={result['per_class']['QUESTION']['f1']:.3f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
