"""
Extract additional question-bearing sentences from an external Kyrgyz text
corpus to address the under-representation of the QUESTION class.

Usage (from repo root):
    python scripts/extract_questions.py \\
        --corpus /path/to/kyrgyz_text_corpus.txt \\
        --existing data/train_data_original.json \\
        --out-questions data/question_data.json \\
        --out-augmented data/train_data_augmented.json \\
        --n 2000 --seed 42

The corpus must be a plain-text UTF-8 file with one sentence per line.
Candidate sentences are filtered to: end with '?', contain 3-30 words,
consist of a single sentence, contain Cyrillic characters, and not already
appear in the existing dataset.
"""

import argparse
import json
import random
import re


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True,
                   help='Path to a Kyrgyz text corpus (one sentence per line, UTF-8).')
    p.add_argument('--existing', default='data/train_data_original.json',
                   help='Existing training JSON (used for dedup).')
    p.add_argument('--out-questions', default='data/question_data.json',
                   help='Output JSON of sampled question sentences only.')
    p.add_argument('--out-augmented', default='data/train_data_augmented.json',
                   help='Output JSON of original + sampled questions (shuffled).')
    p.add_argument('--n', type=int, default=2000, help='Number of questions to sample.')
    p.add_argument('--min-words', type=int, default=3)
    p.add_argument('--max-words', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


args = parse_args()
CORPUS = args.corpus
EXISTING = args.existing
OUTPUT_QUESTIONS = args.out_questions
OUTPUT_AUGMENTED = args.out_augmented
TARGET_QUESTIONS = args.n
MIN_WORDS = args.min_words
MAX_WORDS = args.max_words

random.seed(args.seed)

# ── 1. Load existing sentences for dedup ─────────────────────
with open(EXISTING, 'r') as f:
    existing = json.load(f)

existing_texts = set()
for entry in existing:
    existing_texts.add(entry['text'][0].strip())

print(f"Existing sentences: {len(existing_texts)}")

# ── 2. Extract candidate question sentences ──────────────────
# Keep only lines that END with '?' (sentence-final question mark,
# clean for token-level labeling).
# Filter out: URLs, brackets-heavy noise, non-Cyrillic lines, too short/long.

CYRILLIC_RE = re.compile(r'[А-Яа-яЁёӨөҮүҢңЇї]')
URL_RE = re.compile(r'https?://|www\.')
# Disallow lines with excessive brackets / quotes (often boilerplate/noise).
NOISE_CHARS = set('[]{}<>|\\')

candidates = []
seen = set()

with open(CORPUS, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or not line.endswith('?'):
            continue
        if URL_RE.search(line):
            continue
        if any(c in line for c in NOISE_CHARS):
            continue
        # must contain Cyrillic (some lines are all-numeric / all-latin)
        if not CYRILLIC_RE.search(line):
            continue
        # single sentence: no terminal punctuation mid-line
        body = line[:-1]  # drop final '?'
        if re.search(r'[.!?]\s+[А-ЯA-Z]', body):
            continue

        words = line.split()
        if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
            continue

        # strip surrounding quote-like chars for dedup
        norm = line
        if norm in existing_texts or norm in seen:
            continue
        seen.add(norm)
        candidates.append(line)

print(f"Candidate question sentences extracted: {len(candidates)}")

# ── 3. Sample target amount ──────────────────────────────────
if len(candidates) < TARGET_QUESTIONS:
    print(f"WARNING: only {len(candidates)} candidates available, "
          f"taking all of them.")
    sampled = candidates
else:
    sampled = random.sample(candidates, TARGET_QUESTIONS)

print(f"Sampled: {len(sampled)} questions")

# ── 4. Save question-only file (for reproducibility/inspection) ──
question_entries = [{'text': [s]} for s in sampled]
with open(OUTPUT_QUESTIONS, 'w', encoding='utf-8') as f:
    json.dump(question_entries, f, ensure_ascii=False, indent=2)
print(f"Wrote {OUTPUT_QUESTIONS}")

# ── 5. Save augmented full dataset ───────────────────────────
augmented = existing + question_entries
# Shuffle so questions don't cluster at the end
random.shuffle(augmented)

with open(OUTPUT_AUGMENTED, 'w', encoding='utf-8') as f:
    json.dump(augmented, f, ensure_ascii=False, indent=2)
print(f"Wrote {OUTPUT_AUGMENTED} ({len(augmented)} entries)")

# ── 6. Quick label statistics for sanity check ───────────────
def count_labels(entries):
    counts = {'O': 0, 'PERIOD': 0, 'COMMA': 0, 'QUESTION': 0}
    for e in entries:
        text = e['text'][0]
        for tok in text.split():
            word = tok.rstrip('.,?!;:\'\"“”–—…()[]{}«»')
            if not word:
                continue
            trailing = tok[len(word):]
            lab = 'O'
            for ch in trailing:
                if ch == '.':
                    lab = 'PERIOD'; break
                elif ch == ',':
                    lab = 'COMMA'; break
                elif ch == '?':
                    lab = 'QUESTION'; break
            counts[lab] += 1
    return counts

print("\nLabel counts — original:")
print(count_labels(existing))
print("\nLabel counts — augmented:")
print(count_labels(augmented))

# ── 7. Show 10 random samples for native-speaker sanity check ─
print("\n" + "=" * 60)
print("SAMPLE 10 QUESTIONS — please sanity-check:")
print("=" * 60)
for s in random.sample(sampled, 10):
    print(f"  • {s}")
