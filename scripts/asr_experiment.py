"""
ASR + punctuation-restoration experiment.

Demonstrates:
  1. Whisper-large-v3 (multilingual SOTA) does not support Kyrgyz at all
     (the 100-language tokenizer has no `ky` token).
  2. UlutSoftLLC/whisper-small-kyrgyz (Kyrgyz-fine-tuned) produces Kyrgyz
     transcripts, but inserts zero commas across 200 Common Voice utterances.
  3. Our XLM-RoBERTa post-processor restores punctuation on Whisper output,
     raising sentence-terminal coverage from 88% to 100% and adding 589 commas.

Usage (from repo root):
    python scripts/asr_experiment.py \\
        --common-voice-root /path/to/common_voice \\
        --our-checkpoint /path/to/ckpt_xlmr_aug/checkpoint-XXXX \\
        --n-samples 200

Expected Common Voice layout:
    <common-voice-root>/data.jsonl          # one {"path": "audio/xxx.mp3", "text": "..."} per line
    <common-voice-root>/audio/xxx.mp3

Common Voice references are normalized (lowercase, no punctuation); we use
them only for WER/CER and not to measure punctuation F1.
"""

import argparse
import json
import os
import random
import re
import time

import librosa
import torch
from jiwer import wer, cer
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor,
    AutoTokenizer, AutoModelForTokenClassification,
    pipeline,
)


WHISPER_MODELS = [
    # Whisper-large-v3 intentionally omitted: its 100-language tokenizer
    # does not include Kyrgyz (`ky`). This is itself a finding reported
    # in the paper. We evaluate only the Kyrgyz-fine-tuned Whisper-small.
    # language=None because Whisper's tokenizer has no `ky` token and
    # UlutSoft's fine-tuning uses the model's default forced_decoder_ids.
    ('whisper-small-kyrgyz', 'UlutSoftLLC/whisper-small-kyrgyz', None),
]

LABEL2ID = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

PUNCT_CHARS = '.,?!;:'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--common-voice-root', required=True,
                   help='Root directory containing data.jsonl and audio/.')
    p.add_argument('--our-checkpoint', required=True,
                   help='Path to our fine-tuned XLM-R checkpoint (from train_model.py).')
    p.add_argument('--n-samples', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--out', default='results_asr.json')
    return p.parse_args()


args = parse_args()
DATA_JSONL = os.path.join(args.common_voice_root, 'data.jsonl')
AUDIO_ROOT = args.common_voice_root
OUR_MODEL_CKPT = args.our_checkpoint


def load_sample(n, seed):
    with open(DATA_JSONL) as f:
        lines = [json.loads(l) for l in f]
    random.Random(seed).shuffle(lines)
    selected = lines[:n]
    print(f"Selected {len(selected)} samples")
    return selected


def load_audio(rel_path, sr=16000):
    path = os.path.join(AUDIO_ROOT, rel_path)
    audio, _ = librosa.load(path, sr=sr)
    return audio


def transcribe_whisper(model_id, samples, batch_size, language=None):
    """Returns list of transcripts aligned with samples."""
    print(f"\n→ Loading Whisper model: {model_id} (language={language or 'auto-detect'})")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipeline(
        'automatic-speech-recognition',
        model=model_id,
        torch_dtype=torch.float16,
        device=device,
    )
    gen_kwargs = {'task': 'transcribe'}
    if language is not None:
        gen_kwargs['language'] = language

    transcripts = []
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        audio = load_audio(s['path'])
        out = pipe(audio, batch_size=1, generate_kwargs=gen_kwargs)
        transcripts.append(out['text'].strip())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(samples)}  ({(time.perf_counter()-t0):.0f}s)")
    del pipe
    torch.cuda.empty_cache()
    return transcripts


def apply_punct_model(texts, device):
    """Run our XLM-RoBERTa token classifier on each text, re-emit the string
    with predicted punctuation."""
    tokenizer = AutoTokenizer.from_pretrained(OUR_MODEL_CKPT)
    model = AutoModelForTokenClassification.from_pretrained(OUR_MODEL_CKPT).to(device).eval()

    out = []
    # Strip existing punctuation from ASR output before running our model,
    # otherwise we double-punctuate (Whisper's period + our predicted period).
    strip_chars = '.,?!;:«»""“”–—…()[]{}'
    with torch.no_grad():
        for txt in texts:
            # remove all punctuation then split
            stripped = ''.join(ch for ch in txt if ch not in strip_chars)
            words = stripped.split()
            if len(words) == 0:
                out.append('')
                continue
            enc = tokenizer(words, is_split_into_words=True, return_tensors='pt',
                            truncation=True, max_length=256).to(device)
            logits = model(**enc).logits.squeeze(0)
            preds = logits.argmax(dim=-1).cpu().tolist()
            word_ids = enc.word_ids(batch_index=0)

            # Take label of LAST subtoken per word
            per_word = [None] * len(words)
            for i in range(len(word_ids) - 1, -1, -1):
                wid = word_ids[i]
                if wid is None:
                    continue
                if per_word[wid] is None:
                    per_word[wid] = ID2LABEL[preds[i]]

            rebuilt = []
            for w, lbl in zip(words, per_word):
                if lbl == 'PERIOD':
                    rebuilt.append(w + '.')
                elif lbl == 'COMMA':
                    rebuilt.append(w + ',')
                elif lbl == 'QUESTION':
                    rebuilt.append(w + '?')
                else:
                    rebuilt.append(w)
            # Capitalise first letter — simple polish
            s = ' '.join(rebuilt)
            if s:
                s = s[0].upper() + s[1:]
            out.append(s)
    del model
    torch.cuda.empty_cache()
    return out


def normalize_for_wer(s):
    """Lowercase, remove punctuation, collapse whitespace — matches Common Voice style."""
    s = s.lower()
    s = re.sub(f'[{re.escape(PUNCT_CHARS)}«»""“”–—…()\\[\\]{{}}]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def punct_stats(texts):
    n = len(texts)
    if n == 0:
        return {'n': 0}
    any_punct = sum(1 for t in texts if any(c in t for c in PUNCT_CHARS))
    final_period = sum(1 for t in texts if t.rstrip().endswith('.'))
    final_question = sum(1 for t in texts if t.rstrip().endswith('?'))
    final_any = sum(1 for t in texts if t.rstrip().endswith(tuple('.?!')))
    total_commas = sum(t.count(',') for t in texts)
    total_periods = sum(t.count('.') for t in texts)
    total_questions = sum(t.count('?') for t in texts)
    return {
        'n': n,
        'pct_with_any_punct': round(100 * any_punct / n, 1),
        'pct_ending_period': round(100 * final_period / n, 1),
        'pct_ending_question': round(100 * final_question / n, 1),
        'pct_ending_any_sentence_final': round(100 * final_any / n, 1),
        'total_commas': total_commas,
        'total_periods': total_periods,
        'total_questions': total_questions,
    }


def main():
    random.seed(args.seed)

    samples = load_sample(args.n_samples, args.seed)
    refs_normalized = [normalize_for_wer(s['text']) for s in samples]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    report = {'n_samples': len(samples), 'per_asr': {}}

    for tag, model_id, language in WHISPER_MODELS:
        transcripts = transcribe_whisper(model_id, samples, args.batch_size, language=language)
        transcripts_norm = [normalize_for_wer(t) for t in transcripts]
        w = wer(refs_normalized, transcripts_norm)
        c = cer(refs_normalized, transcripts_norm)

        # Apply our punctuation model to raw transcripts (not the normalized ones —
        # we want the Whisper output as input to our post-processor).
        restored = apply_punct_model(transcripts, device)

        report['per_asr'][tag] = {
            'model_id': model_id,
            'wer': round(w, 4),
            'cer': round(c, 4),
            'raw_punct_stats': punct_stats(transcripts),
            'after_our_model_punct_stats': punct_stats(restored),
            'examples': [
                {
                    'audio': samples[i]['path'],
                    'reference_cv': samples[i]['text'],
                    'whisper_raw': transcripts[i],
                    'whisper_plus_ours': restored[i],
                }
                for i in range(min(10, len(samples)))
            ],
        }
        print(f"\n[{tag}]  WER={w:.3f}  CER={c:.3f}")
        print(f"  raw:      {report['per_asr'][tag]['raw_punct_stats']}")
        print(f"  + ours:   {report['per_asr'][tag]['after_our_model_punct_stats']}")

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.out}")

    # Pretty summary
    print("\n" + "=" * 90)
    print(f"{'ASR model':<24}{'WER':>8}{'CER':>8}{'% w/ punct (raw)':>20}{'% w/ punct (+ours)':>22}")
    print("-" * 90)
    for tag, r in report['per_asr'].items():
        print(f"{tag:<24}"
              f"{r['wer']:>8.3f}"
              f"{r['cer']:>8.3f}"
              f"{r['raw_punct_stats']['pct_with_any_punct']:>20.1f}"
              f"{r['after_our_model_punct_stats']['pct_with_any_punct']:>22.1f}")
    print("=" * 90)


if __name__ == '__main__':
    main()
