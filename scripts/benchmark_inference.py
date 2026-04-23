"""
Inference efficiency benchmark for the three fine-tuned Kyrgyz punctuation
models. Measures: #params, disk size, GPU latency, GPU throughput, CPU latency.

Usage (from repo root, after running train_model.py for all three models):
    python scripts/benchmark_inference.py \\
        --mbert-ckpt ckpt_mbert_aug/checkpoint-XXXX \\
        --xlmr-ckpt  ckpt_xlmr_aug/checkpoint-XXXX \\
        --kbert-ckpt ckpt_kbert_aug/checkpoint-XXXX

The exact checkpoint sub-directory name (checkpoint-XXXX) depends on the
training run length; use the `ckpt_*_aug` best-checkpoint saved by
load_best_model_at_end.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mbert-ckpt', required=True)
    p.add_argument('--xlmr-ckpt',  required=True)
    p.add_argument('--kbert-ckpt', required=True)
    p.add_argument('--out', default='results_benchmark.json')
    return p.parse_args()


args = parse_args()
MODELS = [
    {'tag': 'mBERT',      'ckpt': args.mbert_ckpt},
    {'tag': 'XLM-R',      'ckpt': args.xlmr_ckpt},
    {'tag': 'KyrgyzBERT', 'ckpt': args.kbert_ckpt},
]

# Realistic Kyrgyz test sentence — ~20 tokens, typical length.
SAMPLE = ("Мен мектепке барам аны эс алдырып, эски тапшырмасын карап, жаңы "
          "тапшырма алып, үйгө кайтам")

SEQ_LEN = 128
WARMUP = 10
ITERS = 200
BATCH = 32


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def bench_latency_gpu(model, tokenizer, device, sample, iters=ITERS, warmup=WARMUP):
    """Single-sample forward-pass latency (batch=1)."""
    enc = tokenizer(sample, return_tensors='pt', padding='max_length',
                    max_length=SEQ_LEN, truncation=True).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**enc)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            start.record()
            _ = model(**enc)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
    return np.median(times), np.mean(times), np.std(times)


def bench_throughput_gpu(model, tokenizer, device, sample, batch=BATCH,
                         iters=50, warmup=5):
    """Throughput at batch=32."""
    enc = tokenizer([sample] * batch, return_tensors='pt',
                    padding='max_length', max_length=SEQ_LEN,
                    truncation=True).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**enc)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(**enc)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    total_samples = batch * iters
    return total_samples / dt  # samples/sec


def bench_latency_cpu(model, tokenizer, sample, iters=20, warmup=3):
    """Single-sample CPU forward-pass latency."""
    model = model.to('cpu').eval()
    enc = tokenizer(sample, return_tensors='pt', padding='max_length',
                    max_length=SEQ_LEN, truncation=True)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**enc)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(**enc)
            times.append((time.perf_counter() - t0) * 1000)  # ms
    return np.median(times), np.mean(times), np.std(times)


def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Sequence length: {SEQ_LEN}, batch: {BATCH}, iters: {ITERS}")
    print()

    results = []
    for m in MODELS:
        print(f"→ {m['tag']}  ({m['ckpt']})")
        tokenizer = AutoTokenizer.from_pretrained(m['ckpt'])
        model = AutoModelForTokenClassification.from_pretrained(m['ckpt']).to(device)

        n_params = count_params_m(model)
        disk_mb = dir_size_mb(m['ckpt'])
        lat_med, lat_mean, lat_std = bench_latency_gpu(model, tokenizer, device, SAMPLE)
        throughput = bench_throughput_gpu(model, tokenizer, device, SAMPLE)
        cpu_lat_med, cpu_lat_mean, cpu_lat_std = bench_latency_cpu(model, tokenizer, SAMPLE)

        entry = {
            'model': m['tag'],
            'params_M': round(n_params, 2),
            'disk_MB': round(disk_mb, 1),
            'gpu_latency_ms_median': round(lat_med, 2),
            'gpu_latency_ms_mean': round(lat_mean, 2),
            'gpu_latency_ms_std': round(lat_std, 2),
            'gpu_throughput_samples_per_sec': round(throughput, 1),
            'cpu_latency_ms_median': round(cpu_lat_med, 1),
            'cpu_latency_ms_mean': round(cpu_lat_mean, 1),
            'cpu_latency_ms_std': round(cpu_lat_std, 1),
        }
        results.append(entry)
        print(json.dumps(entry, indent=2))
        print()

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    # Save
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    # Pretty print summary
    print("=" * 90)
    print(f"{'Model':<12}{'Params':>10}{'Disk':>10}{'GPU lat':>12}{'GPU thru':>14}{'CPU lat':>12}")
    print(f"{'':<12}{'(M)':>10}{'(MB)':>10}{'(ms, bs=1)':>12}{'(samples/s)':>14}{'(ms, bs=1)':>12}")
    print("-" * 90)
    for r in results:
        print(f"{r['model']:<12}"
              f"{r['params_M']:>10.1f}"
              f"{r['disk_MB']:>10.1f}"
              f"{r['gpu_latency_ms_median']:>12.2f}"
              f"{r['gpu_throughput_samples_per_sec']:>14.1f}"
              f"{r['cpu_latency_ms_median']:>12.1f}")
    print("=" * 90)

    # Relative to XLM-R (our best-accuracy baseline)
    xlmr = next(r for r in results if r['model'] == 'XLM-R')
    print(f"\nRelative to XLM-R (accuracy champion):")
    for r in results:
        rel_params = r['params_M'] / xlmr['params_M']
        rel_lat = r['gpu_latency_ms_median'] / xlmr['gpu_latency_ms_median']
        rel_thru = r['gpu_throughput_samples_per_sec'] / xlmr['gpu_throughput_samples_per_sec']
        print(f"  {r['model']:<12} params: {rel_params:.2f}x  gpu_latency: {rel_lat:.2f}x  throughput: {rel_thru:.2f}x")


if __name__ == '__main__':
    main()
