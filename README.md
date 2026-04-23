# Punctuation Restoration for Kyrgyz Language

Code, data, and results for the paper
**"Punctuation Restoration for Kyrgyz Language: A Comparative Study of Multilingual Transformer Models"**
(Uvalieva & Muhametjanova, under revision at *ACM Transactions on Asian and Low-Resource Language Information Processing* — manuscript ID `TALLIP-26-0124`).

This repository accompanies the revised submission and contains everything needed to reproduce the three-way model comparison (mBERT, XLM-RoBERTa, KyrgyzBERT), the question-mark data-augmentation ablation, the inference-efficiency benchmark, the zero-shot KyrgyzBERT evaluation, and the ASR-integration experiment with Whisper.

## Task

Given an unpunctuated Kyrgyz token sequence, assign to each token one of four labels — `O`, `PERIOD`, `COMMA`, `QUESTION` — indicating the punctuation mark that should follow it. The three non-`O` marks cover 100% of sentence-final positions and 98.6% of non-paired punctuation tokens in our corpus (see Section 2.3 of the paper for the full Kyrgyz punctuation inventory).

## Main results

All numbers are **weighted F1** on the 15% held-out test split of the **augmented** 16,028-sentence corpus (24,179 tokens).

| Model                 | Precision | Recall  | F1        | Params   |
|-----------------------|-----------|---------|-----------|----------|
| Rule-based baseline   | 0.773     | 0.765   | 0.768     | —        |
| KyrgyzBERT            | 0.920     | 0.923   | 0.921     | 35.6 M   |
| mBERT                 | 0.929     | 0.932   | 0.930     | 177.3 M  |
| **XLM-RoBERTa-base**  | **0.943** | **0.944** | **0.944** | 277.5 M |

Per-class F1 (XLM-RoBERTa): `O` 0.968, `COMMA` 0.767, `PERIOD` 0.972, `QUESTION` 0.830.

### Question-mark augmentation ablation

| Model       | Original F1 | Augmented F1 | Δ        |
|-------------|-------------|--------------|----------|
| mBERT       | 0.643       | 0.809        | +0.166   |
| XLM-RoBERTa | 0.704       | 0.830        | +0.126   |
| KyrgyzBERT  | 0.410       | 0.738        | +0.328   |

### Zero-shot KyrgyzBERT (MLM mask-filling, no fine-tuning)

Weighted F1 0.769; `PERIOD` F1 0.001; `QUESTION` F1 0.000. The zero-shot model essentially never predicts a sentence-terminal mark because the pre-training distribution is dominated by commas. See Section 5.1 of the paper.

### Inference efficiency

| Model       | Params  | Disk    | GPU latency (bs=1) | GPU throughput (bs=32) | CPU latency (bs=1) |
|-------------|---------|---------|--------------------|------------------------|--------------------|
| mBERT       | 177.3 M | 2029 MB | 3.69 ms            | 1,032 samples/s        | 35.2 ms            |
| XLM-R       | 277.5 M | 3175 MB | 3.71 ms            | 1,043 samples/s        | 35.0 ms            |
| KyrgyzBERT  | **35.6 M** | **407 MB** | **1.29 ms** | **4,495 samples/s** | **9.5 ms** |

Measured on a single NVIDIA RTX 5080. KyrgyzBERT is 7.8× smaller, 2.9× faster on GPU, and 3.7× faster on CPU than XLM-RoBERTa — a favorable trade-off for deployment on constrained hardware despite the 2.3-point F1 gap.

### ASR-integration experiment

On 200 random Common Voice Kyrgyz utterances (seed 42):

| Metric                                    | Whisper-small-kyrgyz | +our XLM-R post-processor |
|-------------------------------------------|----------------------|----------------------------|
| % utterances with any punctuation         | 88.0                 | 100.0                      |
| % utterances ending in `.`                | 85.0                 | 100.0                      |
| Total commas across 200 utterances        | **0**                | **589**                    |

Whisper-large-v3 is *not* included because its tokenizer does not contain a Kyrgyz (`ky`) language token; calling `model.generate(language="ky")` raises `ValueError: Unsupported language: ky`. See Section 5.4 of the paper.

## Repository layout

```
.
├── README.md                          — you are here
├── main.tex                           — LaTeX source of the revised manuscript
├── references.bib                     — BibTeX references
│
├── data/
│   ├── train_data_original.json       — initial 14,028-sentence corpus
│   ├── train_data_augmented.json      — augmented 16,028-sentence corpus
│   └── question_data.json             — the 2,000 sampled question sentences
│
├── scripts/
│   ├── train_model.py                 — unified fine-tuning (all three models)
│   ├── extract_questions.py           — builds train_data_augmented.json
│   ├── zero_shot_kyrgyzbert.py        — zero-shot MLM mask-fill evaluation
│   ├── benchmark_inference.py         — inference-efficiency benchmark
│   └── asr_experiment.py              — Whisper → our XLM-R post-processor pipeline
│
├── results/
│   ├── results_{mbert,xlmr,kbert}_{orig,aug}.json   — per-model metrics + confusion
│   ├── results_kbert_zeroshot.json    — zero-shot KyrgyzBERT metrics
│   ├── results_benchmark.json         — inference-efficiency numbers
│   └── results_asr.json               — ASR-integration metrics + 10 examples
│
├── figures/
│   ├── confusion_matrix_mbert_aug.png
│   ├── confusion_matrix_xlmr_aug.png
│   └── confusion_matrix_kbert_aug.png
│
├── mbert_baseline.ipynb               — original-submission training notebook
└── xlmr_finetune.ipynb                — original-submission training notebook
```

Model checkpoints (large binaries) are hosted on Hugging Face, not in this repo:

- XLM-RoBERTa (augmented): https://huggingface.co/Zarinaaa/xlmr-kyrgyz-punctuation
- mBERT (augmented): https://huggingface.co/Zarinaaa/mbert-kyrgyz-punctuation
- KyrgyzBERT (augmented): trained locally with `train_model.py`; release planned.

## Reproducing the results

### Requirements

```
python        >= 3.10
torch         >= 2.0   (CUDA recommended for training)
transformers  >= 4.40
scikit-learn
numpy
matplotlib
librosa       (asr_experiment.py only)
jiwer         (asr_experiment.py only)
```

### 1. Fine-tune all three models

```bash
# augmented-data runs (main results)
python scripts/train_model.py --model bert-base-multilingual-cased \
    --data data/train_data_augmented.json --tag mbert_aug
python scripts/train_model.py --model xlm-roberta-base \
    --data data/train_data_augmented.json --tag xlmr_aug
python scripts/train_model.py --model metinovadilet/KyrgyzBert \
    --data data/train_data_augmented.json --tag kbert_aug

# original-data runs (for the Table 10 ablation)
python scripts/train_model.py --model bert-base-multilingual-cased \
    --data data/train_data_original.json --tag mbert_orig
# ... analogously for xlmr_orig, kbert_orig
```

Each run produces `ckpt_<tag>/`, `confusion_matrix_<tag>.png`, and `results_<tag>.json` in the current working directory. Expected wall-clock time on a single RTX 5080: about 1 min for KyrgyzBERT, 3 min for mBERT, 4 min for XLM-RoBERTa.

### 2. (Optional) Re-create the augmented dataset from scratch

```bash
python scripts/extract_questions.py \
    --corpus /path/to/kyrgyz_text_corpus.txt \
    --existing data/train_data_original.json \
    --out-questions data/question_data.json \
    --out-augmented data/train_data_augmented.json
```

The corpus is a plain-text UTF-8 file with one sentence per line. Under seed 42 the resulting `train_data_augmented.json` is byte-identical to the file shipped in this repo.

### 3. Zero-shot KyrgyzBERT

```bash
python scripts/zero_shot_kyrgyzbert.py \
    --data data/train_data_augmented.json \
    --out results/results_kbert_zeroshot.json
```

### 4. Inference-efficiency benchmark

```bash
python scripts/benchmark_inference.py \
    --mbert-ckpt ckpt_mbert_aug/checkpoint-XXXX \
    --xlmr-ckpt  ckpt_xlmr_aug/checkpoint-XXXX \
    --kbert-ckpt ckpt_kbert_aug/checkpoint-XXXX \
    --out results/results_benchmark.json
```

### 5. ASR-integration experiment

Requires a local copy of Common Voice Kyrgyz arranged as:
```
<cv-root>/data.jsonl            # {"path": "audio/xxx.mp3", "text": "..."} per line
<cv-root>/audio/xxx.mp3
```

```bash
python scripts/asr_experiment.py \
    --common-voice-root /path/to/common_voice \
    --our-checkpoint ckpt_xlmr_aug/checkpoint-XXXX \
    --n-samples 200 \
    --out results/results_asr.json
```

## Citation

```bibtex
@article{uvalieva2026kyrgyz,
  author  = {Uvalieva, Zarina and Muhametjanova, Gulshat},
  title   = {Punctuation Restoration for {Kyrgyz} Language: A Comparative Study of Multilingual Transformer Models},
  journal = {ACM Transactions on Asian and Low-Resource Language Information Processing},
  year    = {2026},
  note    = {Under revision}
}
```

## Contact

- Zarina Uvalieva — <zarina.uvalievaa@gmail.com>
- Gulshat Muhametjanova — <gulshat.muhametjanova@manas.edu.kg>
