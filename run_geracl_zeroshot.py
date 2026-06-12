#!/usr/bin/env python3
"""Zero-shot тональность GeRaCl в отдельном env (transformers>=4.49), multi-GPU.

Читает тексты из parquet (в порядке строк), шардирует по всем видимым GPU
(каждый процесс грузит свою копию модели), считает вероятности классов
[negative, neutral, positive] тем же способом, что geracl-ветка
_build_zero_shot_predictor в dynamic_topics_sentiments.py (get_similarities +
softmax по меткам), и сохраняет матрицу (n_docs, 3) в .npy + мета-JSON.

Дальше основной пайплайн (env topic-sentiment) подхватывает файл через
--zero-shot-precomputed и усредняет с классификаторами ансамбля.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import multiprocessing as mp

import numpy as np
import pandas as pd

CANONICAL = ["negative", "neutral", "positive"]


def normalize_label(s: str) -> str:
    t = str(s).strip().lower()
    if "позитив" in t or "positive" in t:
        return "positive"
    if "негатив" in t or "negative" in t or "отриц" in t:
        return "negative"
    if "нейтр" in t or "neutral" in t:
        return "neutral"
    return t


def _predict_shard(job: tuple) -> tuple[int, np.ndarray]:
    """Воркер: своя копия модели на своём GPU, возвращает (shard_idx, probs)."""
    (shard_idx, device_str, model_name, labels, docs, batch_size, log_every) = job
    import torch
    from transformers import AutoTokenizer
    from geracl import GeraclHF, ZeroShotClassificationPipeline

    model = GeraclHF.from_pretrained(model_name).to(device_str).eval()
    if hasattr(model, "config"):
        setattr(model.config, "device", device_str)
    core = getattr(model, "_classification_core", None)
    if core is not None and hasattr(core, "_device"):
        core._device = device_str
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    zpipe = ZeroShotClassificationPipeline(model, tokenizer, device=device_str, progress_bar=False)

    label_to_canon_idx = {}
    for li, lab in enumerate(labels):
        canon = normalize_label(lab)
        if canon not in CANONICAL:
            raise SystemExit(f"Метка {lab!r} не отображается в канонические {CANONICAL}")
        label_to_canon_idx[li] = CANONICAL.index(canon)

    n = len(docs)
    out = np.zeros((n, len(CANONICAL)), dtype=np.float32)
    t0 = time.time()
    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        lo, hi = b * batch_size, min(n, (b + 1) * batch_size)
        batch = docs[lo:hi]
        with torch.no_grad():
            sims = zpipe.get_similarities(batch, labels, same_labels=True, batch_size=len(batch))
        row_i = lo
        for sim in sims:
            probs = torch.softmax(sim.view(-1, len(labels)), dim=1).detach().cpu()
            for row_probs in probs.tolist():
                for li, prob in enumerate(row_probs):
                    out[row_i, label_to_canon_idx[li]] = float(prob)
                row_i += 1
        if row_i != hi:
            raise SystemExit(f"shard {shard_idx}: несовпадение строк в батче {b}: {row_i} != {hi}")
        if (b + 1) % log_every == 0 or b + 1 == n_batches:
            print(f"[shard {shard_idx} @ {device_str}] {b+1}/{n_batches} batches, {time.time()-t0:.0f}s", flush=True)
    return shard_idx, out


def main() -> None:
    p = argparse.ArgumentParser(description="GeRaCl zero-shot probs -> npy (multi-GPU)")
    p.add_argument("--in-parquet", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--out-npy", required=True)
    p.add_argument("--model", default="deepvk/GeRaCl-USER2-base")
    p.add_argument("--labels", nargs="+", default=["нейтральный", "позитивный", "негативный"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--devices",
        nargs="+",
        default=["auto"],
        help="'auto' = все видимые GPU; иначе список вида cuda:0 cuda:1 (или cpu).",
    )
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    import torch

    if args.devices == ["auto"]:
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]
    else:
        devices = list(args.devices)

    df = pd.read_parquet(args.in_parquet, columns=[args.text_col])
    docs = df[args.text_col].astype(str).tolist()
    n = len(docs)
    print(f"docs: {n} from {args.in_parquet}; devices: {devices}", flush=True)

    # Непрерывные шарды в порядке строк — итог склеивается без перестановок.
    n_dev = len(devices)
    bounds = [round(i * n / n_dev) for i in range(n_dev + 1)]
    jobs = []
    for i, dev in enumerate(devices):
        lo, hi = bounds[i], bounds[i + 1]
        if hi > lo:
            jobs.append((i, dev, args.model, list(args.labels), docs[lo:hi], args.batch_size, args.log_every))

    t0 = time.time()
    if len(jobs) == 1:
        results = [_predict_shard(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_predict_shard, jobs))

    results.sort(key=lambda x: x[0])
    out = np.concatenate([r[1] for r in results], axis=0)
    if out.shape[0] != n:
        raise SystemExit(f"Итоговое число строк {out.shape[0]} != {n}")

    out_path = Path(args.out_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    meta = {
        "model": args.model,
        "labels": list(args.labels),
        "canonical_order": CANONICAL,
        "n_docs": n,
        "in_parquet": str(args.in_parquet),
        "text_col": args.text_col,
        "devices": devices,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved: {out_path} shape={out.shape} in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
