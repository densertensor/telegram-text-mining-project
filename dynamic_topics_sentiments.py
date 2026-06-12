from __future__ import annotations


import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize




def _prepare_torch_runtime_env() -> None:
   # В долгоживущих tmux-сессиях текущая директория может стать невалидной
   # (удалена/переименована), из-за чего импорт torch/transformers падает.
   try:
       os.getcwd()
   except FileNotFoundError:
       os.chdir(str(Path.home()))


   debug_dir = Path.home() / ".cache" / "torch_compile_debug"
   debug_dir.mkdir(parents=True, exist_ok=True)
   os.environ.setdefault("TORCH_COMPILE_DEBUG_DIR", str(debug_dir))




_prepare_torch_runtime_env()


import torch
from transformers import AutoTokenizer, pipeline
from tqdm.auto import tqdm




CANONICAL_SENTIMENT_LABELS = ["negative", "neutral", "positive"]
DEFAULT_SENTIMENT_MODELS = [
   "cardiffnlp/twitter-xlm-roberta-base-sentiment",
   "seara/rubert-base-cased-russian-sentiment",
]
DEFAULT_ZERO_SHOT_SENTIMENT_MODEL = "deepvk/GeRaCl-USER2-base"
DEFAULT_ZERO_SHOT_LABELS = ["нейтральный", "позитивный", "негативный"]
FILTER_QUERY_STOP_TERMS = [
   "президент",
   "президента",
   "президенту",
   "президентом",
   "президенте",
   "путин",
   "путина",
   "путину",
   "путиным",
   "путине",
   "владимир",
   "владимира",
   "владимиру",
   "владимиром",
   "владимирович",
   "владимировича",
   "владимировичу",
   "верховный",
]




def _parse_args() -> argparse.Namespace:
   p = argparse.ArgumentParser(
       description=(
           "Динамический анализ BERTopic для president_putin_selection JSONL "
           "с эмбеддингами deepvk/USER-bge-m3."
       )
   )
   p.add_argument(
       "--in-jsonl",
       default="outputs/president_putin_selection_20260423_233414.jsonl",
       help="Входной JSONL с отфильтрованными постами.",
   )
   p.add_argument(
       "--out-dir",
       default="topic_model_outputs_dynamic_putin_user_bge",
       help="Каталог для артефактов модели, таблиц и графиков.",
   )
   p.add_argument("--text-col", default="text")
   p.add_argument("--date-col", default="date")
   p.add_argument("--embedding-model-name", default="deepvk/USER-bge-m3")
   p.add_argument(
       "--embeddings-path",
       default=None,
       help="Необязательный путь к кэшу эмбеддингов (.npy). Если не задан, используется out-dir.",
   )
   p.add_argument("--batch-size", type=int, default=128)
   p.add_argument("--min-topic-size", type=int, default=100)
   p.add_argument("--min-samples", type=int, default=3)
   p.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="leaf")
   p.add_argument("--nr-topics", default="auto", help="'auto' или целое число.")
   p.add_argument("--random-state", type=int, default=42)
   p.add_argument(
       "--extra-stop-terms",
       nargs="+",
       default=[],
       help="Дополнительные стоп-термины (словоформы поискового запроса корпуса).",
   )
   p.add_argument("--vectorizer-min-df", type=int, default=10)
   p.add_argument("--vectorizer-max-df", type=float, default=0.9)
   p.add_argument(
       "--nr-bins",
       type=int,
       default=50,
       help="Количество временных бинов для эволюции тем, если --sentiment-bin-days <= 1.",
   )
   p.add_argument(
       "--no-evolution",
       action="store_true",
       help=(
           "Отключить расчёт эволюции слов тем через нативный BERTopic "
           "topics_over_time (evolution_tuning)."
       ),
   )
   p.add_argument("--top-n-dynamic", type=int, default=5, help="Сколько топиков показывать на графиках динамики.")
   p.add_argument("--top-n-delta", type=int, default=20, help="Сколько топиков показывать по дельте доли (до/после).")
   p.add_argument(
       "--no-interactive-topic-plots",
       action="store_true",
       help="Отключить интерактивные HTML-графики динамики топиков.",
   )
   p.add_argument("--analysis-start", default="2022-02-24", help="Дата начала анализа (включительно).")
   p.add_argument("--analysis-end", default=None, help="Дата конца анализа (включительно); по умолчанию максимальная дата в данных.")
   p.add_argument("--cutoff-date", default="2026-01-16", help="Дата отсечки для сценариев до/после.")
   p.add_argument("--no-multi-process-embeddings", action="store_true")
   p.add_argument(
       "--gpu-ids",
       nargs="+",
       type=int,
       default=None,
       help=(
           "ID GPU для использования. По умолчанию скрипт берёт последние N GPU "
           "(см. --last-n-gpus)."
       ),
   )
   p.add_argument(
       "--last-n-gpus",
       type=int,
       default=3,
       help="Сколько последних GPU использовать по умолчанию, если --gpu-ids не задан.",
   )
   p.add_argument(
       "--umap-backend",
       choices=["auto", "cuml", "umap-learn"],
       default="umap-learn",
       help="Использовать RAPIDS cuML UMAP, если доступен; иначе umap-learn.",
   )
   p.add_argument("--umap-n-neighbors", type=int, default=100, help="Параметр UMAP n_neighbors.")
   p.add_argument("--umap-n-components", type=int, default=5, help="Параметр UMAP n_components.")
   p.add_argument("--umap-min-dist", type=float, default=0.0, help="Параметр UMAP min_dist.")
   p.add_argument("--umap-metric", default="cosine", help="Метрика расстояния UMAP.")
   p.add_argument("--dpi", type=int, default=300)
   p.add_argument(
       "--sentiment-models",
       nargs="+",
       default=DEFAULT_SENTIMENT_MODELS,
       help="Классификационные модели тональности для ансамбля.",
   )
   p.add_argument(
       "--zero-shot-sentiment-model",
       default=DEFAULT_ZERO_SHOT_SENTIMENT_MODEL,
       help="Zero-shot модель тональности для ансамбля.",
   )
   p.add_argument(
       "--zero-shot-precomputed",
       default=None,
       help=(
           "Путь к .npy (n_docs, 3) с готовыми вероятностями zero-shot модели "
           "в порядке [negative, neutral, positive] и в порядке строк входного "
           "корпуса (см. run_geracl_zeroshot.py, отдельный env geracl). "
           "Если задан, zero-shot модель в процессе не загружается."
       ),
   )
   p.add_argument(
       "--zero-shot-labels",
       nargs="+",
       default=DEFAULT_ZERO_SHOT_LABELS,
       help="Кандидатные метки для zero-shot тональности.",
   )
   p.add_argument(
       "--zero-shot-hypothesis-template",
       default="Это сообщение {}.",
       help="Шаблон гипотезы для zero-shot пайплайна тональности.",
   )
   p.add_argument("--sentiment-batch-size", type=int, default=128)
   p.add_argument("--sentiment-max-length", type=int, default=512)
   p.add_argument(
       "--sentiment-parallel-backend",
       choices=["process", "thread"],
       default="process",
       help="Параллельный backend для инференса тональности по моделям.",
   )
   p.add_argument(
       "--sentiment-rolling-days",
       type=int,
       default=7,
       help=(
           "Окно сглаживания динамики тональности в календарных днях, "
           "применяется к дневным данным (0 или 1 — без сглаживания)."
       ),
   )
   p.add_argument(
       "--sentiment-exclude-mixed",
       action="store_true",
       help=(
           "Исключить из графика динамики тональности посты с флагом _mixed "
           "(Путин/президент РФ вместе с иностранным лидером)."
       ),
   )
   p.add_argument(
       "--sentiment-bin-days",
       type=int,
       default=30,
       help=(
           "Размер календарного бина в днях для CSV тональности и динамики тем. "
           "На сам график тональности не влияет — он строится по дневным долям."
       ),
   )
   p.add_argument(
       "--disable-sentiment",
       action="store_true",
       help="Пропустить ансамблевый инференс тональности и графики тональности.",
   )
   p.add_argument(
       "--force-recompute",
       action="store_true",
       help=(
           "Принудительно полностью пересчитать BERTopic/тональность, даже если "
           "кэшированные артефакты уже есть в --out-dir."
       ),
   )
   p.add_argument(
       "--render-only",
       action="store_true",
       help=(
           "Пересобрать только визуализации из существующих артефактов в --out-dir "
           "(без запуска BERTopic/моделей тональности)."
       ),
   )
   p.add_argument(
       "--recompute-sentiment-only",
       action="store_true",
       help=(
           "Пересчитать только тональность из кэшированных docs_with_topics/topic_info, "
           "без повторного запуска BERTopic/эмбеддингов."
       ),
   )
   p.add_argument(
       "--recompute-topics-only",
       action="store_true",
       help=(
           "Пересчитать только BERTopic/topics-over-time, переиспользуя кэш эмбеддингов "
           "без расчёта тональности."
       ),
   )
   return p.parse_args()




def _load_russian_stopwords() -> list[str] | None:
   try:
       import nltk
       from nltk.corpus import stopwords


       try:
           return sorted(set(stopwords.words("russian")))
       except LookupError:
           nltk.download("stopwords")
           return sorted(set(stopwords.words("russian")))
   except Exception as exc:
       print(f"Could not load NLTK Russian stopwords, continuing without them: {exc}")
       return None




def _build_topic_stopwords(extra_terms: list[str] | None = None) -> list[str]:
   base = _load_russian_stopwords() or []
   extra = {t.strip().lower() for t in (extra_terms or []) if str(t).strip()}
   merged = sorted(set(base) | set(FILTER_QUERY_STOP_TERMS) | extra)
   print(
       f"Topic stop-terms: {len(merged)} total "
       f"({len(FILTER_QUERY_STOP_TERMS)} filter-query terms, {len(extra)} extra)."
   )
   return merged




def _parse_nr_topics(value: str) -> str | int:
   value = str(value).strip()
   if value.lower() == "auto":
       return "auto"
   return int(value)




def _normalize_sentiment_label(s: str) -> str:
   t = str(s).strip().lower()
   if "позитив" in t or "positive" in t:
       return "positive"
   if "негатив" in t or "negative" in t or "отриц" in t:
       return "negative"
   if "нейтр" in t or "neutral" in t:
       return "neutral"
   return t




def _empty_class_scores(n: int) -> dict[str, list[float]]:
   return {label: [0.0] * n for label in CANONICAL_SENTIMENT_LABELS}




def _effective_max_length(pipe_obj, requested: int) -> int:
   model_obj = getattr(pipe_obj, "model", None)
   if model_obj is None:
       return requested
   cfg = getattr(model_obj, "config", None)
   cmax = getattr(cfg, "max_position_embeddings", None)
   if isinstance(cmax, int) and cmax > 0:
       return min(requested, cmax)
   return requested




def _predict_cls_batch(pipe_obj, batch: list[str], max_length: int) -> dict[str, list[float]]:
   preds = pipe_obj(batch, truncation=True, max_length=max_length, top_k=None)
   if isinstance(preds, dict):
       preds = [preds]
   class_scores = _empty_class_scores(len(batch))
   for i, item in enumerate(preds):
       entries = [item] if isinstance(item, dict) else list(item)
       for pred in entries:
           label = _normalize_sentiment_label(pred["label"])
           score = float(pred["score"])
           if label in class_scores:
               class_scores[label][i] = score
   return class_scores




def _import_geracl():
   try:
       from geracl import GeraclHF, ZeroShotClassificationPipeline
       return GeraclHF, ZeroShotClassificationPipeline
   except Exception as first_exc:
       # Когда cwd — каталог, рядом с которым лежит клон geracl,
       # `import geracl` может резолвиться в директорию репозитория как namespace
       # пакет (без __init__.py), который не экспортирует GeraclHF.
       sys.modules.pop("geracl", None)
       candidates: list[Path] = []
       env_path = os.environ.get("GERACL_PATH")
       if env_path:
           candidates.append(Path(env_path).expanduser())
       # Ожидаемый layout: клон geracl лежит рядом с корнем репозитория
       candidates.append(Path(__file__).resolve().parents[1] / "geracl")

       for candidate in candidates:
           package_init = candidate / "geracl" / "__init__.py"
           if not package_init.exists():
               continue
           cand_str = str(candidate)
           if cand_str not in sys.path:
               sys.path.insert(0, cand_str)
           try:
               sys.modules.pop("geracl", None)
               from geracl import GeraclHF, ZeroShotClassificationPipeline
               return GeraclHF, ZeroShotClassificationPipeline
           except Exception:
               continue

       raise RuntimeError(
           "Could not import local 'geracl' package required for GeRaCl zero-shot model. "
           "Install it in env (`pip install -e <path-to-geracl-clone>`) "
           "or set GERACL_PATH to the cloned repo root."
       ) from first_exc




def _build_zero_shot_predictor(
   model_name: str,
   labels: list[str],
   hypothesis_template: str,
   batch_size: int,
   device: int,
):
   model_name_norm = model_name.strip().lower()
   if model_name_norm.startswith("deepvk/geracl"):
       GeraclHF, ZeroShotClassificationPipeline = _import_geracl()
       device_str = f"cuda:{device}" if device >= 0 else "cpu"
       model = GeraclHF.from_pretrained(model_name).to(device_str).eval()
       # В core GeRaCl есть собственное поле device, которое может остаться "cuda:0"
       # из чекпоинта/конфига; принудительно синхронизируем его с выбранным устройством.
       if hasattr(model, "config"):
           setattr(model.config, "device", device_str)
       core = getattr(model, "_classification_core", None)
       if core is not None and hasattr(core, "_device"):
           core._device = device_str
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       zpipe = ZeroShotClassificationPipeline(
           model,
           tokenizer,
           device=device_str,
           progress_bar=False,
       )


       def _predict(batch: list[str]) -> dict[str, list[float]]:
           sims = zpipe.get_similarities(
               batch,
               labels,
               same_labels=True,
               batch_size=len(batch),
           )
           class_scores = _empty_class_scores(len(batch))
           row_i = 0
           for sim in sims:
               probs = torch.softmax(sim.view(-1, len(labels)), dim=1).detach().cpu()
               for row_probs in probs.tolist():
                   for label_idx, prob in enumerate(row_probs):
                       canonical = _normalize_sentiment_label(labels[label_idx])
                       if canonical in class_scores:
                           class_scores[canonical][row_i] = float(prob)
                   row_i += 1
           return class_scores


       return _predict


   zpipe = pipeline(
       "zero-shot-classification",
       model=model_name,
       device=device,
       trust_remote_code=True,
   )


   def _predict(batch: list[str]) -> dict[str, list[float]]:
       outputs = zpipe(
           batch,
           candidate_labels=labels,
           hypothesis_template=hypothesis_template,
           multi_label=False,
           batch_size=batch_size,
       )
       if isinstance(outputs, dict):
           outputs = [outputs]
       class_scores = _empty_class_scores(len(outputs))
       for i, item in enumerate(outputs):
           for raw_label, raw_score in zip(item.get("labels", []), item.get("scores", [])):
               canonical = _normalize_sentiment_label(raw_label)
               if canonical in class_scores:
                   class_scores[canonical][i] = float(raw_score)
       return class_scores


   return _predict




def _resolve_gpu_ids(explicit_ids: list[int] | None, last_n: int = 3) -> list[int]:
   if not torch.cuda.is_available():
       return []


   n_gpus = torch.cuda.device_count()
   if n_gpus <= 0:
       return []


   if explicit_ids:
       seen: set[int] = set()
       resolved: list[int] = []
       for raw_idx in explicit_ids:
           idx = int(raw_idx)
           if idx < 0 or idx >= n_gpus:
               raise SystemExit(
                   f"Invalid GPU id {idx}. Available GPU ids: 0..{n_gpus - 1}."
               )
           if idx in seen:
               continue
           seen.add(idx)
           resolved.append(idx)
       return resolved


   take_n = max(1, int(last_n))
   start = max(0, n_gpus - take_n)
   return list(range(start, n_gpus))




def _run_sentiment_model_probs(
   docs: list[str],
   model_name: str,
   batch_size: int,
   max_length: int,
   device: int,
   tqdm_position: int = 0,
) -> np.ndarray:
   n_docs = len(docs)
   label_to_idx = {k: i for i, k in enumerate(CANONICAL_SENTIMENT_LABELS)}
   probs = np.zeros((n_docs, len(CANONICAL_SENTIMENT_LABELS)), dtype=np.float32)
   spipe = pipeline("sentiment-analysis", model=model_name, device=device, top_k=None)
   eff_len = _effective_max_length(spipe, max_length)
   total_batches = (n_docs + batch_size - 1) // batch_size
   progress_desc = f"sent-cls[{device}] {model_name.split('/')[-1]}"
   for start in tqdm(
       range(0, n_docs, batch_size),
       total=total_batches,
       desc=progress_desc,
       unit="batch",
       position=tqdm_position,
       leave=False,
   ):
       batch = docs[start : start + batch_size]
       class_scores = _predict_cls_batch(spipe, batch, max_length=eff_len)
       rows = np.zeros((len(batch), len(CANONICAL_SENTIMENT_LABELS)), dtype=np.float32)
       for label, scores in class_scores.items():
           rows[:, label_to_idx[label]] = np.asarray(scores, dtype=np.float32)
       probs[start : start + len(batch)] = rows
   return probs




def _run_zero_shot_model_probs(
   docs: list[str],
   model_name: str,
   labels: list[str],
   hypothesis_template: str,
   batch_size: int,
   device: int,
   tqdm_position: int = 0,
) -> np.ndarray | None:
   n_docs = len(docs)
   label_to_idx = {k: i for i, k in enumerate(CANONICAL_SENTIMENT_LABELS)}
   probs = np.zeros((n_docs, len(CANONICAL_SENTIMENT_LABELS)), dtype=np.float32)
   try:
       zs_predict = _build_zero_shot_predictor(
           model_name=model_name,
           labels=labels,
           hypothesis_template=hypothesis_template,
           batch_size=batch_size,
           device=device,
       )
   except Exception as exc:
       print(
           f"--- Warning: zero-shot model '{model_name}' unavailable on device {device}: {exc}. "
           "Skipping this model in ensemble. ---"
       )
       return None
   total_batches = (n_docs + batch_size - 1) // batch_size
   progress_desc = f"sent-zs[{device}] {model_name.split('/')[-1]}"
   for start in tqdm(
       range(0, n_docs, batch_size),
       total=total_batches,
       desc=progress_desc,
       unit="batch",
       position=tqdm_position,
       leave=False,
   ):
       batch = docs[start : start + batch_size]
       class_scores = zs_predict(batch)
       rows = np.zeros((len(batch), len(CANONICAL_SENTIMENT_LABELS)), dtype=np.float32)
       for label, scores in class_scores.items():
           if label in label_to_idx:
               rows[:, label_to_idx[label]] = np.asarray(scores, dtype=np.float32)
       probs[start : start + len(batch)] = rows
   return probs




def _run_sentiment_ensemble(
   docs: list[str],
   sentiment_models: list[str],
   zero_shot_model: str,
   zero_shot_labels: list[str],
   zero_shot_hypothesis_template: str,
   batch_size: int,
   max_length: int,
   gpu_ids: list[int] | None = None,
   parallel_backend: str = "process",
   zero_shot_precomputed: np.ndarray | None = None,
) -> pd.DataFrame:
   n_docs = len(docs)
   if n_docs == 0:
       return pd.DataFrame(
           columns=[
               "ensemble_negative_prob",
               "ensemble_neutral_prob",
               "ensemble_positive_prob",
               "ensemble_label",
               "ensemble_score",
           ]
       )


   sum_scores = np.zeros((n_docs, len(CANONICAL_SENTIMENT_LABELS)), dtype=np.float32)
   model_count = 0
   label_to_idx = {k: i for i, k in enumerate(CANONICAL_SENTIMENT_LABELS)}


   selected_gpus = list(gpu_ids or [])
   if not selected_gpus:
       default_gpu = 0 if torch.cuda.is_available() else -1
       selected_gpus = [default_gpu]


   jobs: list[tuple[str, str, int]] = []
   gpu_pos = 0
   for model_name in sentiment_models:
       if not str(model_name).strip():
           continue
       device = selected_gpus[gpu_pos % len(selected_gpus)]
       jobs.append(("classification", model_name, device))
       gpu_pos += 1


   if zero_shot_precomputed is not None:
       if zero_shot_precomputed.shape != (n_docs, len(CANONICAL_SENTIMENT_LABELS)):
           raise SystemExit(
               "zero-shot-precomputed shape mismatch: "
               f"{zero_shot_precomputed.shape} != ({n_docs}, {len(CANONICAL_SENTIMENT_LABELS)}). "
               "Файл должен быть посчитан по тому же корпусу в том же порядке строк."
           )
       sum_scores += zero_shot_precomputed.astype(np.float32)
       model_count += 1
       print("--- Sentiment model (zero-shot): precomputed probs loaded ---")
   elif zero_shot_model.strip():
       device = selected_gpus[gpu_pos % len(selected_gpus)]
       jobs.append(("zero_shot", zero_shot_model, device))


   if not jobs:
       raise SystemExit("No sentiment models configured for ensemble.")


   max_workers = min(len(jobs), max(1, len(selected_gpus)))
   if parallel_backend == "process":
       executor_cls = ProcessPoolExecutor
       executor_kwargs = {"mp_context": mp.get_context("spawn")}
   else:
       executor_cls = ThreadPoolExecutor
       executor_kwargs = {}


   with executor_cls(max_workers=max_workers, **executor_kwargs) as ex:
       futures = []
       for bar_pos, (kind, model_name, device) in enumerate(jobs):
           if kind == "classification":
               print(f"--- Sentiment model (classification): {model_name} on device {device} ---")
               fut = ex.submit(
                   _run_sentiment_model_probs,
                   docs,
                   model_name,
                   batch_size,
                   max_length,
                   device,
                   bar_pos,
               )
           else:
               print(f"--- Sentiment model (zero-shot): {model_name} on device {device} ---")
               fut = ex.submit(
                   _run_zero_shot_model_probs,
                   docs,
                   model_name,
                   zero_shot_labels,
                   zero_shot_hypothesis_template,
                   batch_size,
                   device,
                   bar_pos,
               )
           futures.append(fut)


       for fut in as_completed(futures):
           probs = fut.result()
           if probs is None:
               continue
           sum_scores += probs
           model_count += 1


   if model_count == 0:
       raise SystemExit(
           "No sentiment models could be executed for ensemble. "
           "Check model availability and environment."
       )


   ensemble_probs = sum_scores / float(model_count)
   best_idx = ensemble_probs.argmax(axis=1)
   labels = [CANONICAL_SENTIMENT_LABELS[i] for i in best_idx]
   scores = ensemble_probs[np.arange(n_docs), best_idx]


   return pd.DataFrame(
       {
           "ensemble_negative_prob": ensemble_probs[:, label_to_idx["negative"]],
           "ensemble_neutral_prob": ensemble_probs[:, label_to_idx["neutral"]],
           "ensemble_positive_prob": ensemble_probs[:, label_to_idx["positive"]],
           "ensemble_label": labels,
           "ensemble_score": scores,
       }
   )




def _safe_embedding_cache_path(out_dir: Path, embedding_model_name: str) -> Path:
   safe_name = embedding_model_name.replace("/", "_").replace(":", "_")
   return out_dir / f"embeddings_{safe_name}.npy"




def _build_umap_model(
   backend: str,
   random_state: int,
   n_neighbors: int,
   n_components: int,
   min_dist: float,
   metric: str,
):
   if backend in {"auto", "cuml"}:
       try:
           from cuml.manifold import UMAP as CumlUMAP


           print("Using cuML UMAP backend.")
           return CumlUMAP(
               n_neighbors=n_neighbors,
               n_components=n_components,
               min_dist=min_dist,
               metric=metric,
               random_state=random_state,
           )
       except Exception as exc:
           if backend == "cuml":
               raise
           print(f"cuML UMAP unavailable, falling back to umap-learn: {exc}")


   from umap import UMAP


   print("Using umap-learn UMAP backend.")
   return UMAP(
       n_neighbors=n_neighbors,
       n_components=n_components,
       min_dist=min_dist,
       metric=metric,
       random_state=random_state,
   )




def _compute_or_load_embeddings(
   docs: list[str],
   embedding_model: SentenceTransformer,
   embeddings_path: Path,
   batch_size: int,
   use_multi_process: bool,
   embedding_devices: list[str] | None = None,
) -> np.ndarray:
   if embeddings_path.exists():
       print(f"--- Loading existing embeddings from {embeddings_path} ---")
       embeddings = np.load(embeddings_path)
       if embeddings.shape[0] != len(docs):
           raise ValueError(
               f"Cached embeddings row count mismatch: embeddings={embeddings.shape[0]}, docs={len(docs)}. "
               f"Delete {embeddings_path} or use a different --embeddings-path."
           )
       return embeddings


   print(f"--- Computing embeddings on {len(docs)} documents ---")
   if use_multi_process:
       pool_kwargs = {}
       if embedding_devices:
           pool_kwargs["target_devices"] = embedding_devices
       pool = embedding_model.start_multi_process_pool(**pool_kwargs)
       try:
           embeddings = embedding_model.encode(
               docs,
               pool=pool,
               batch_size=batch_size,
               show_progress_bar=True,
           )
       finally:
           embedding_model.stop_multi_process_pool(pool)
   else:
       embeddings = embedding_model.encode(
           docs,
           batch_size=batch_size,
           show_progress_bar=True,
           normalize_embeddings=True,
       )


   embeddings = normalize(np.asarray(embeddings), norm="l2")
   print(f"--- Saving embeddings to {embeddings_path} ---")
   np.save(embeddings_path, embeddings)
   return embeddings




def _load_jsonl(in_path: Path, text_col: str, date_col: str) -> pd.DataFrame:
   rows: list[dict] = []
   with in_path.open("r", encoding="utf-8") as f:
       for i, line in enumerate(f, start=1):
           line = line.strip()
           if not line:
               continue
           try:
               obj = json.loads(line)
           except json.JSONDecodeError:
               continue
           obj["_row_id"] = i
           rows.append(obj)


   if not rows:
       raise SystemExit("Input JSONL has no valid rows.")


   df = pd.DataFrame(rows)
   if text_col not in df.columns:
       raise SystemExit(f"Text column not found: {text_col!r}")
   if date_col not in df.columns:
       raise SystemExit(f"Date column not found: {date_col!r}")


   df[text_col] = (
       df[text_col]
       .astype(str)
       .str.replace(r"\s+", " ", regex=True)
       .str.strip()
   )
   df = df[df[text_col].notna() & df[text_col].ne("") & df[text_col].ne("nan")].copy()
   df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
   df = df[df[date_col].notna()].copy()
   return df




@dataclass
class ScenarioWindow:
   name: str
   before_start: pd.Timestamp
   before_end: pd.Timestamp
   after_start: pd.Timestamp
   after_end: pd.Timestamp
   n_days_after: int




def _build_scenarios(
   analysis_start: pd.Timestamp,
   analysis_end: pd.Timestamp,
   cutoff_date: pd.Timestamp,
) -> list[ScenarioWindow]:
   cutoff_day = cutoff_date.normalize()
   before_end = cutoff_day - pd.Timedelta(days=1)


   if before_end < analysis_start:
       raise SystemExit("Cutoff is earlier than analysis start; cannot build 'before' window.")
   if cutoff_day > analysis_end:
       raise SystemExit("Cutoff is later than analysis end; cannot build 'after' window.")


   n_days_after = int((analysis_end - cutoff_day).days) + 1


   fixed = ScenarioWindow(
       name="v1_fixed_2022_to_cutoff",
       before_start=analysis_start,
       before_end=before_end,
       after_start=cutoff_day,
       after_end=analysis_end,
       n_days_after=n_days_after,
   )


   symmetric_before_start = before_end - pd.Timedelta(days=n_days_after - 1)
   symmetric = ScenarioWindow(
       name="v2_symmetric_n_days",
       before_start=symmetric_before_start,
       before_end=before_end,
       after_start=cutoff_day,
       after_end=analysis_end,
       n_days_after=n_days_after,
   )
   return [fixed, symmetric]




def _topic_lookup(topic_model: BERTopic) -> pd.DataFrame:
   info = topic_model.get_topic_info().copy()
   info["Topic"] = pd.to_numeric(info["Topic"], errors="coerce").astype("Int64")
   info = info.rename(columns={"Topic": "topic_id", "Count": "topic_count", "Name": "topic_name"})
   return info




def _build_before_after_table(
   df_docs: pd.DataFrame,
   topic_lookup: pd.DataFrame,
   scenario: ScenarioWindow,
) -> pd.DataFrame:
   before_mask = (df_docs["date_day"] >= scenario.before_start) & (df_docs["date_day"] <= scenario.before_end)
   after_mask = (df_docs["date_day"] >= scenario.after_start) & (df_docs["date_day"] <= scenario.after_end)


   before = (
       df_docs[before_mask]
       .groupby("topic_id", dropna=False)
       .size()
       .rename("before_n")
       .reset_index()
   )
   after = (
       df_docs[after_mask]
       .groupby("topic_id", dropna=False)
       .size()
       .rename("after_n")
       .reset_index()
   )
   merged = before.merge(after, on="topic_id", how="outer").fillna(0)
   merged["before_n"] = merged["before_n"].astype(int)
   merged["after_n"] = merged["after_n"].astype(int)


   total_before = max(1, int(merged["before_n"].sum()))
   total_after = max(1, int(merged["after_n"].sum()))
   merged["before_share"] = merged["before_n"] / total_before
   merged["after_share"] = merged["after_n"] / total_after
   merged["delta_share"] = merged["after_share"] - merged["before_share"]
   merged["abs_delta_share"] = merged["delta_share"].abs()


   merged = merged.merge(
       topic_lookup[["topic_id", "topic_name"]],
       on="topic_id",
       how="left",
   )
   merged["scenario"] = scenario.name
   merged["before_start"] = scenario.before_start.date().isoformat()
   merged["before_end"] = scenario.before_end.date().isoformat()
   merged["after_start"] = scenario.after_start.date().isoformat()
   merged["after_end"] = scenario.after_end.date().isoformat()
   merged["n_days_after"] = scenario.n_days_after
   return merged.sort_values("abs_delta_share", ascending=False)




def _clean_topic_name_for_display(name: str) -> str:
   s = str(name).strip()
   head, sep, tail = s.partition("_")
   if sep and head.lstrip("-").isdigit():
       return tail.strip()
   return s


def _build_topics_over_time_from_docs(df_docs: pd.DataFrame, bin_days: int = 0) -> pd.DataFrame:
   required = {"date_day", "topic_id"}
   if not required.issubset(df_docs.columns):
       return pd.DataFrame()

   tmp = df_docs.copy()
   tmp["date_day"] = pd.to_datetime(tmp["date_day"], errors="coerce", utc=True).dt.normalize()
   tmp["topic_id"] = pd.to_numeric(tmp["topic_id"], errors="coerce").astype("Int64")
   tmp = tmp[tmp["date_day"].notna() & tmp["topic_id"].notna()].copy()
   if tmp.empty:
       return pd.DataFrame()
   tmp["topic_id"] = tmp["topic_id"].astype(int)

   words_map: dict[int, str] = {}
   if "topic_name" in tmp.columns:
       topic_meta = tmp[["topic_id", "topic_name"]].drop_duplicates(subset=["topic_id"], keep="first")
       words_map = {
           int(tid): _clean_topic_name_for_display(name)
           for tid, name in zip(topic_meta["topic_id"], topic_meta["topic_name"], strict=False)
       }

   daily = (
       tmp.groupby(["date_day", "topic_id"], dropna=False)
       .size()
       .rename("Frequency")
       .reset_index()
       .rename(columns={"date_day": "Timestamp", "topic_id": "Topic"})
   )
   daily["Topic"] = daily["Topic"].astype(int)

   if bin_days and bin_days > 1:
       out = (
           daily.groupby(
               ["Topic", pd.Grouper(key="Timestamp", freq=f"{int(bin_days)}D")],
               dropna=False,
           )["Frequency"]
           .sum()
           .reset_index()
       )
   else:
       out = daily.copy()

   out = out.sort_values(["Timestamp", "Topic"]).reset_index(drop=True)
   out["Words"] = out["Topic"].map(lambda t: words_map.get(int(t), ""))
   return out


def _compute_topics_over_time_evolution(
   topic_model: BERTopic | None,
   model_dir: Path,
   df_docs: pd.DataFrame,
   text_col: str,
   bin_days: int,
   nr_bins: int,
) -> pd.DataFrame:
   """Нативный BERTopic topics_over_time с evolution_tuning: топ-слова каждой
   темы пересчитываются по c-TF-IDF в каждом временном бине и сглаживаются
   с предыдущим бином и глобальным представлением темы."""
   required = {text_col, "date_day", "topic_id"}
   if not required.issubset(df_docs.columns):
       return pd.DataFrame()

   if topic_model is None:
       if not model_dir.exists():
           print(f"BERTopic model dir is missing, skipping word evolution: {model_dir}")
           return pd.DataFrame()
       print(f"--- Loading BERTopic model for word evolution: {model_dir} ---")
       topic_model = BERTopic.load(str(model_dir))

   tmp = df_docs[[text_col, "date_day", "topic_id"]].copy()
   tmp["date_day"] = pd.to_datetime(tmp["date_day"], errors="coerce", utc=True).dt.normalize()
   tmp["topic_id"] = pd.to_numeric(tmp["topic_id"], errors="coerce")
   tmp = tmp[tmp["date_day"].notna() & tmp["topic_id"].notna()].copy()
   if tmp.empty:
       return pd.DataFrame()

   docs = tmp[text_col].astype(str).tolist()
   topics = tmp["topic_id"].astype(int).tolist()

   if bin_days and bin_days > 1:
       # Те же календарные бины, что и в _build_topics_over_time_from_docs.
       base = tmp["date_day"].min()
       bin_idx = (tmp["date_day"] - base).dt.days // int(bin_days)
       timestamps = list(base + pd.to_timedelta(bin_idx * int(bin_days), unit="D"))
       nr_bins_arg = None
   else:
       timestamps = tmp["date_day"].tolist()
       nr_bins_arg = max(2, int(nr_bins))

   print(f"--- Computing topic word evolution on {len(docs)} docs ---")
   evolution = topic_model.topics_over_time(
       docs,
       timestamps,
       topics=topics,
       nr_bins=nr_bins_arg,
       evolution_tuning=True,
       global_tuning=True,
   )
   evolution["Timestamp"] = pd.to_datetime(evolution["Timestamp"], errors="coerce", utc=True)
   return evolution


def _plot_dynamic_top_topics(
   topics_over_time: pd.DataFrame,
   topic_lookup: pd.DataFrame,
   out_path: Path,
   top_n: int,
   title: str,
   dpi: int,
   keep_topics: list[int] | None = None,
   cutoff_dates: list[pd.Timestamp] | None = None,
) -> None:
   if topics_over_time.empty:
       return
   df = topics_over_time.copy()
   df = df[~df["Topic"].isin([-1, 0])].copy()
   if df.empty:
       return


   top_topics = keep_topics or _select_top_topics_by_abs_delta_share(df, top_n=top_n)
   df = df[df["Topic"].isin(top_topics)].copy()
   if df.empty:
       return


   name_map = (
       topic_lookup.set_index("topic_id")["topic_name"].to_dict()
       if "topic_name" in topic_lookup.columns
       else {}
   )
   pivot = (
       df.pivot_table(index="Timestamp", columns="Topic", values="Frequency", aggfunc="sum", fill_value=0.0)
       .sort_index()
   )
   if pivot.empty:
       return


   out_path.parent.mkdir(parents=True, exist_ok=True)
   fig, ax = plt.subplots(figsize=(14, 7))
   for tid in pivot.columns:
       label = str(name_map.get(tid, "")).strip() or str(tid)
       ax.plot(pivot.index, pivot[tid], linewidth=1.4, label=label[:95])
   for idx, c in enumerate(cutoff_dates or []):
       label = f"Отсечка {pd.to_datetime(c, utc=True).date().isoformat()}" if idx == 0 else None
       ax.axvline(
           pd.to_datetime(c, utc=True),
           color="#111111",
           linestyle="--",
           linewidth=1.2,
           alpha=0.75,
           label=label,
       )
   ax.set_title(title)
   ax.set_xlabel("Дата")
   ax.set_ylabel("Частота темы")
   ax.grid(alpha=0.25)
   ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)
   fig.tight_layout()
   fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
   plt.close(fig)




def _select_top_topics_by_total_frequency(df: pd.DataFrame, top_n: int) -> list[int]:
   if df.empty:
       return []
   topic_totals = (
       df.groupby("Topic", dropna=False)["Frequency"]
       .sum()
       .sort_values(ascending=False)
   )
   return topic_totals.head(max(1, top_n)).index.tolist()




def _select_top_topics_by_abs_delta_share(df: pd.DataFrame, top_n: int) -> list[int]:
   if df.empty:
       return []
   if "Timestamp" not in df.columns:
       return _select_top_topics_by_total_frequency(df, top_n=top_n)


   pivot = (
       df.pivot_table(index="Timestamp", columns="Topic", values="Frequency", aggfunc="sum", fill_value=0.0)
       .sort_index()
   )
   if pivot.empty:
       return []
   if len(pivot.index) < 2:
       return _select_top_topics_by_total_frequency(df, top_n=top_n)


   # Сравниваем доли топиков в первом и последнем бине (аналог до/после).
   start_share = pivot.iloc[0] / max(1.0, float(pivot.iloc[0].sum()))
   end_share = pivot.iloc[-1] / max(1.0, float(pivot.iloc[-1].sum()))
   abs_delta = (end_share - start_share).abs().sort_values(ascending=False)
   return abs_delta.head(max(1, top_n)).index.tolist()




def _plot_dynamic_top_topics_interactive(
   topics_over_time: pd.DataFrame,
   topic_lookup: pd.DataFrame,
   out_path: Path,
   top_n: int,
   title: str,
   keep_topics: list[int] | None = None,
   cutoff_dates: list[pd.Timestamp] | None = None,
) -> None:
   if topics_over_time.empty:
       return


   try:
       import plotly.express as px
       import plotly.graph_objects as go
   except Exception as exc:
       print(f"Plotly is unavailable, skipping interactive topic plot {out_path.name}: {exc}")
       return


   df = topics_over_time.copy()
   if "Topic" not in df.columns or "Frequency" not in df.columns:
       print(f"topics_over_time lacks required columns for interactive plot: {out_path.name}")
       return


   if "Timestamp" in df.columns:
       df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
       df = df[df["Timestamp"].notna()].copy()
   else:
       print(f"topics_over_time has no Timestamp column, skipping interactive plot: {out_path.name}")
       return


   df["Topic"] = pd.to_numeric(df["Topic"], errors="coerce").astype("Int64")
   df = df[df["Topic"].notna()].copy()
   df = df[~df["Topic"].isin([-1, 0])].copy()
   if df.empty:
       return


   top_topics = keep_topics or _select_top_topics_by_abs_delta_share(df, top_n=top_n)
   df = df[df["Topic"].isin(top_topics)].copy()
   if df.empty:
       return


   if "Words" not in df.columns:
       df["Words"] = ""


   # Оставляем ровно один ряд на пару (топик, timestamp).
   df = (
       df.sort_values(["Timestamp", "Topic"])
       .groupby(["Timestamp", "Topic"], as_index=False)
       .agg(
           Frequency=("Frequency", "sum"),
           Words=("Words", "first"),
       )
   )
   if df.empty:
       return


   name_map = (
       topic_lookup.set_index("topic_id")["topic_name"].to_dict()
       if "topic_name" in topic_lookup.columns
       else {}
   )
   label_map = {
       int(topic_id): (str(name_map.get(int(topic_id), "")).strip() or str(int(topic_id)))
       for topic_id in top_topics
   }
   df["topic_label"] = df["Topic"].map(lambda t: label_map.get(int(t), f"{int(t)}"))


   fig = px.line(
       df,
       x="Timestamp",
       y="Frequency",
       color="topic_label",
       markers=False,
       title=title,
       hover_data={
           "Topic": True,
           "Words": True,
           "Frequency": True,
           "Timestamp": True,
           "topic_label": False,
       },
   )
   fig.update_layout(
       template="plotly_white",
       xaxis_title="Дата",
       yaxis_title="Частота",
       legend_title_text="Топик",
       hovermode="closest",
   )
   ymax = float(df["Frequency"].max()) if not df.empty else 0.0
   for idx, c in enumerate(cutoff_dates or []):
       c_ts = pd.to_datetime(c, utc=True)
       fig.add_trace(
           go.Scatter(
               x=[c_ts, c_ts],
               y=[0.0, ymax],
               mode="lines",
               name=(f"Отсечка {c_ts.date().isoformat()}" if idx == 0 else "Отсечка"),
               line=dict(color="#111111", dash="dash", width=1.2),
               opacity=0.75,
               hoverinfo="skip",
               showlegend=True,
           )
       )
   out_path.parent.mkdir(parents=True, exist_ok=True)
   fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)




def _plot_before_after_delta(
   comparison_df: pd.DataFrame,
   out_path: Path,
   top_n: int,
   title: str,
   dpi: int,
) -> None:
   if comparison_df.empty:
       return
   df = comparison_df[~comparison_df["topic_id"].isin([-1, 0])].copy()
   if df.empty:
       return
   df = df.sort_values("abs_delta_share", ascending=False).head(top_n).copy()
   df = df.iloc[::-1]


   def _clean_topic_name(name: str) -> str:
       # Названия BERTopic часто начинаются с "<topic_id>_..."; убираем этот префикс для отображения.
       s = str(name).strip()
       head, sep, tail = s.partition("_")
       if sep and head.isdigit():
           return tail.strip()
       return s

   labels = [_clean_topic_name(str(n))[:60] for n in df["topic_name"]]
   out_path.parent.mkdir(parents=True, exist_ok=True)
   fig, ax = plt.subplots(figsize=(13, max(6, 0.42 * len(df))))
   colors = ["#1f77b4" if v >= 0 else "#d62728" for v in df["delta_share"]]
   ax.barh(labels, df["delta_share"], color=colors, alpha=0.9)
   ax.axvline(0, color="black", linewidth=1.0, alpha=0.8)
   ax.set_title(title)
   ax.set_xlabel("Изменение доли темы (после - до)")
   ax.grid(axis="x", alpha=0.25)
   fig.tight_layout()
   fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
   plt.close(fig)




def _plot_sentiment_dynamics(
   df_docs: pd.DataFrame,
   out_path: Path,
   title: str,
   cutoff_dates: list[pd.Timestamp],
   dpi: int,
   rolling_days: int,
   bin_days: int = 0,
) -> pd.DataFrame:
   if df_docs.empty or "ensemble_label" not in df_docs.columns:
       return pd.DataFrame()


   daily = (
       df_docs.groupby(["date_day", "ensemble_label"], dropna=False)
       .size()
       .rename("n")
       .reset_index()
   )
   counts = (
       daily.pivot_table(index="date_day", columns="ensemble_label", values="n", fill_value=0.0)
       .sort_index()
   )
   for col in ["positive", "negative", "neutral"]:
       if col not in counts.columns:
           counts[col] = 0.0
   counts = counts[["negative", "neutral", "positive"]].copy()


   # CSV с агрегатами по календарным бинам — для отчётных таблиц;
   # сам график строится по дневным долям, чтобы биннинг не прятал динамику.
   if bin_days and bin_days > 1:
       binned = counts.resample(f"{int(bin_days)}D").sum()
   else:
       binned = counts
   binned_totals = binned.sum(axis=1).replace(0.0, np.nan)
   binned_share = binned.div(binned_totals, axis=0).fillna(0.0)


   out_csv = out_path.with_suffix(".csv")
   out_csv.parent.mkdir(parents=True, exist_ok=True)
   binned_share.to_csv(out_csv)


   # Дневные доли без какого-либо сглаживания.
   daily_totals = counts.sum(axis=1).replace(0.0, np.nan)
   daily_share = counts.div(daily_totals, axis=0)


   # Окно сглаживания задано в календарных днях и применяется к дневным счётчикам,
   # а не к строкам бинированного фрейма; доля взвешена по числу постов в окне.
   smoothed = bool(rolling_days and rolling_days > 1)
   if smoothed:
       roll_counts = counts.rolling(f"{int(rolling_days)}D", min_periods=1).sum()
       roll_totals = roll_counts.sum(axis=1).replace(0.0, np.nan)
       smooth_share = roll_counts.div(roll_totals, axis=0)
   else:
       smooth_share = daily_share


   out_path.parent.mkdir(parents=True, exist_ok=True)
   fig, ax = plt.subplots(figsize=(14, 7))
   if smoothed:
       # Полупрозрачный фон — сырые дневные значения, поверх — сглаженная линия.
       ax.plot(daily_share.index, daily_share["positive"], color="#2ca02c", linewidth=0.8, alpha=0.25)
       ax.plot(daily_share.index, daily_share["negative"], color="#d62728", linewidth=0.8, alpha=0.25)
   pos_label = f"Позитив (скользящее {int(rolling_days)} дн.)" if smoothed else "Позитив"
   neg_label = f"Негатив (скользящее {int(rolling_days)} дн.)" if smoothed else "Негатив"
   ax.plot(smooth_share.index, smooth_share["positive"], color="#2ca02c", linewidth=1.8, label=pos_label)
   ax.plot(smooth_share.index, smooth_share["negative"], color="#d62728", linewidth=1.8, label=neg_label)


   for idx, c in enumerate(cutoff_dates):
       label = f"Отсечка {pd.to_datetime(c, utc=True).date().isoformat()}" if idx == 0 else None
       ax.axvline(c, color="#111111", linestyle="--", linewidth=1.2, alpha=0.75, label=label)


   ax.set_title(title)
   ax.set_xlabel("Дата")
   ax.set_ylabel("Доля постов")
   ax.grid(alpha=0.25)
   ax.legend()
   fig.tight_layout()
   fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
   plt.close(fig)
   return binned_share.reset_index()




def main() -> None:
   args = _parse_args()
   if args.recompute_topics_only and args.render_only:
       raise SystemExit("--recompute-topics-only cannot be used with --render-only.")
   if args.recompute_topics_only and args.recompute_sentiment_only:
       raise SystemExit("--recompute-topics-only cannot be used with --recompute-sentiment-only.")
   if args.recompute_topics_only and args.force_recompute:
       raise SystemExit("--recompute-topics-only cannot be used with --force-recompute.")
   if args.recompute_sentiment_only and args.render_only:
       raise SystemExit("--recompute-sentiment-only cannot be used with --render-only.")
   if args.recompute_sentiment_only and args.force_recompute:
       raise SystemExit("--recompute-sentiment-only cannot be used with --force-recompute.")
   if args.recompute_sentiment_only and args.disable_sentiment:
       raise SystemExit("--recompute-sentiment-only requires sentiment enabled (remove --disable-sentiment).")


   out_dir = Path(args.out_dir)
   out_dir.mkdir(parents=True, exist_ok=True)


   embeddings_path = (
       Path(args.embeddings_path)
       if args.embeddings_path is not None
       else _safe_embedding_cache_path(out_dir, args.embedding_model_name)
   )


   df = _load_jsonl(Path(args.in_jsonl), args.text_col, args.date_col)
   df["date_day"] = df[args.date_col].dt.tz_convert("UTC").dt.normalize()


   analysis_start = pd.Timestamp(args.analysis_start, tz="UTC").normalize()
   analysis_end = (
       pd.Timestamp(args.analysis_end, tz="UTC").normalize()
       if args.analysis_end
       else df["date_day"].max()
   )
   cutoff_date = pd.Timestamp(args.cutoff_date, tz="UTC").normalize()

   zs_precomputed = None
   if args.zero_shot_precomputed:
       zs_precomputed = np.load(args.zero_shot_precomputed)
       print(
           f"--- Loaded precomputed zero-shot probs: {args.zero_shot_precomputed} "
           f"shape={zs_precomputed.shape} ---"
       )


   df = df[(df["date_day"] >= analysis_start) & (df["date_day"] <= analysis_end)].copy()
   if df.empty:
       raise SystemExit("No documents in selected analysis period.")


   docs = df[args.text_col].tolist()
   timestamps = df["date_day"].tolist()
   print(f"Documents for modeling: {len(docs)}")
   print(f"Period: {analysis_start.date()} .. {analysis_end.date()}")


   gpu_ids = _resolve_gpu_ids(args.gpu_ids, args.last_n_gpus)
   if gpu_ids:
       print(f"Using GPUs: {gpu_ids}")
   else:
       print("No CUDA GPUs available; using CPU.")
   embedding_devices = [f"cuda:{idx}" for idx in gpu_ids]


   scenarios = _build_scenarios(analysis_start, analysis_end, cutoff_date)
   cutoff_dates = [cutoff_date]


   docs_with_topics_path = out_dir / "docs_with_topics.parquet"
   docs_with_topics_sent_path = out_dir / "docs_with_topics_and_sentiment.parquet"
   topic_info_path = out_dir / "topic_info.csv"
   topics_over_time_full_path = out_dir / "topics_over_time_full.csv"
   topics_evolution_path = out_dir / "topics_over_time_evolution.csv"
   bertopic_model_dir = out_dir / "bertopic_user_bge_m3_dynamic"


   has_cached_core = docs_with_topics_path.exists() and topic_info_path.exists()
   if args.render_only and not has_cached_core:
       raise SystemExit(
           "render-only requested but cached artifacts are missing: "
           "need docs_with_topics.parquet and topic_info.csv in --out-dir."
       )
   if args.recompute_sentiment_only and not has_cached_core:
       raise SystemExit(
           "recompute-sentiment-only requested but cached core artifacts are missing: "
           "need docs_with_topics.parquet and topic_info.csv in --out-dir."
       )


   use_cached = has_cached_core and not args.force_recompute
   if args.render_only:
       use_cached = True
   if args.recompute_sentiment_only:
       use_cached = True
   if args.recompute_topics_only:
       use_cached = False


   topic_model = None
   topic_info: pd.DataFrame
   df_topics: pd.DataFrame


   if use_cached:
       if args.recompute_sentiment_only:
           print("--- Using cached topic artifacts; recomputing sentiment only ---")
           df_topics = pd.read_parquet(docs_with_topics_path)
       else:
           print("--- Using cached modeling artifacts; skipping BERTopic/sentiment inference ---")
           if docs_with_topics_sent_path.exists() and not args.disable_sentiment:
               df_topics = pd.read_parquet(docs_with_topics_sent_path)
           else:
               df_topics = pd.read_parquet(docs_with_topics_path)
       topic_info = pd.read_csv(topic_info_path)
       if "date_day" in df_topics.columns:
           df_topics["date_day"] = pd.to_datetime(df_topics["date_day"], errors="coerce", utc=True).dt.normalize()
       else:
           if args.date_col not in df_topics.columns:
               raise SystemExit("Cached docs_with_topics* missing date columns.")
           df_topics[args.date_col] = pd.to_datetime(df_topics[args.date_col], errors="coerce", utc=True)
           df_topics["date_day"] = df_topics[args.date_col].dt.normalize()
       df_topics = df_topics[
           (df_topics["date_day"] >= analysis_start) & (df_topics["date_day"] <= analysis_end)
       ].copy()


       if args.recompute_sentiment_only:
           sentiment_cols = [
               "ensemble_negative_prob",
               "ensemble_neutral_prob",
               "ensemble_positive_prob",
               "ensemble_label",
               "ensemble_score",
           ]
           drop_cols = [c for c in sentiment_cols if c in df_topics.columns]
           if drop_cols:
               df_topics = df_topics.drop(columns=drop_cols)
           sent_df = _run_sentiment_ensemble(
               docs=df_topics[args.text_col].astype(str).tolist(),
               sentiment_models=[m for m in args.sentiment_models if str(m).strip()],
               zero_shot_model=args.zero_shot_sentiment_model,
               zero_shot_labels=[x for x in args.zero_shot_labels if str(x).strip()],
               zero_shot_hypothesis_template=args.zero_shot_hypothesis_template,
               batch_size=args.sentiment_batch_size,
               max_length=args.sentiment_max_length,
               gpu_ids=gpu_ids,
               parallel_backend=args.sentiment_parallel_backend,
               zero_shot_precomputed=zs_precomputed,
           )
           df_topics = pd.concat(
               [df_topics.reset_index(drop=True), sent_df.reset_index(drop=True)],
               axis=1,
           )
           df_topics.to_parquet(docs_with_topics_sent_path, index=False)
           print("--- Saved recomputed sentiment to docs_with_topics_and_sentiment.parquet ---")
   else:
       if args.recompute_topics_only and not embeddings_path.exists():
           raise SystemExit(
               "recompute-topics-only requested but cached embeddings are missing: "
               f"{embeddings_path}. Run a full pass first to create embeddings cache."
           )
       vectorizer_model = CountVectorizer(
           stop_words=_build_topic_stopwords(args.extra_stop_terms),
           ngram_range=(1, 2),
           min_df=args.vectorizer_min_df,
           max_df=args.vectorizer_max_df,
       )
       embedding_device = embedding_devices[0] if embedding_devices else "cpu"
       embedding_model = SentenceTransformer(args.embedding_model_name, device=embedding_device)
       embeddings = _compute_or_load_embeddings(
           docs=docs,
           embedding_model=embedding_model,
           embeddings_path=embeddings_path,
           batch_size=args.batch_size,
           use_multi_process=not args.no_multi_process_embeddings,
           embedding_devices=embedding_devices,
       )


       umap_model = _build_umap_model(
           backend=args.umap_backend,
           random_state=args.random_state,
           n_neighbors=args.umap_n_neighbors,
           n_components=args.umap_n_components,
           min_dist=args.umap_min_dist,
           metric=args.umap_metric,
       )
       hdbscan_model = HDBSCAN(
           min_cluster_size=args.min_topic_size,
           min_samples=args.min_samples,
           metric="euclidean",
           cluster_selection_method=args.cluster_selection_method,
           prediction_data=True,
       )


       topic_model = BERTopic(
           embedding_model=embedding_model,
           vectorizer_model=vectorizer_model,
           umap_model=umap_model,
           hdbscan_model=hdbscan_model,
           representation_model=KeyBERTInspired(),
           nr_topics=_parse_nr_topics(args.nr_topics),
           calculate_probabilities=False,
           verbose=True,
       )


       topics, _ = topic_model.fit_transform(docs, embeddings)
       topic_info = _topic_lookup(topic_model)


       df_topics = df.copy()
       df_topics["topic_id"] = topics
       topic_name_map = topic_info.set_index("topic_id")["topic_name"].to_dict()
       df_topics["topic_name"] = df_topics["topic_id"].map(topic_name_map)
       # Сохраняем базовые артефакты topic model до шага расчёта тональности.
       df_topics.to_parquet(docs_with_topics_path, index=False)
       topic_info.to_csv(topic_info_path, index=False)
       print("--- Saved docs_with_topics/topic_info before sentiment inference ---")


       if not args.disable_sentiment and not args.recompute_topics_only:
           # Этап эмбеддингов/BERTopic завершён; очищаем allocator перед загрузкой моделей тональности.
           if torch.cuda.is_available():
               torch.cuda.empty_cache()
           sent_df = _run_sentiment_ensemble(
               docs=docs,
               sentiment_models=[m for m in args.sentiment_models if str(m).strip()],
               zero_shot_model=args.zero_shot_sentiment_model,
               zero_shot_labels=[x for x in args.zero_shot_labels if str(x).strip()],
               zero_shot_hypothesis_template=args.zero_shot_hypothesis_template,
               batch_size=args.sentiment_batch_size,
               max_length=args.sentiment_max_length,
               gpu_ids=gpu_ids,
               parallel_backend=args.sentiment_parallel_backend,
               zero_shot_precomputed=zs_precomputed,
           )
           df_topics = pd.concat(
               [df_topics.reset_index(drop=True), sent_df.reset_index(drop=True)],
               axis=1,
           )
           df_topics.to_parquet(docs_with_topics_sent_path, index=False)


   if not args.disable_sentiment and "ensemble_label" in df_topics.columns:
       df_sent_plot = df_topics
       sent_title = "Sentiment dynamics (positive/negative) - full period"
       if args.sentiment_exclude_mixed:
           if "_mixed" in df_topics.columns:
               mixed_mask = df_topics["_mixed"].fillna(False).astype(bool)
               df_sent_plot = df_topics[~mixed_mask]
               sent_title += " — без mixed-постов"
               print(
                   f"--- Sentiment plot excludes mixed posts: "
                   f"{int(mixed_mask.sum()):,} removed, {len(df_sent_plot):,} kept ---"
               )
           else:
               print("--- --sentiment-exclude-mixed set, but no _mixed column; using full corpus ---")
       _plot_sentiment_dynamics(
           df_docs=df_sent_plot,
           out_path=out_dir / "fig_sentiment_dynamics_full.png",
           title=sent_title,
           cutoff_dates=cutoff_dates,
           dpi=args.dpi,
           rolling_days=args.sentiment_rolling_days,
           bin_days=args.sentiment_bin_days,
       )


   # Глобальный динамический анализ тем с тем же календарным биннингом,
   # что и в графике тональности (sentiment_bin_days).
   topics_over_time_full = _build_topics_over_time_from_docs(
       df_docs=df_topics,
       bin_days=args.sentiment_bin_days,
   )
   if not topics_over_time_full.empty:
       topics_over_time_full.to_csv(topics_over_time_full_path, index=False)
   elif topics_over_time_full_path.exists():
       topics_over_time_full = pd.read_csv(topics_over_time_full_path)
       if "Timestamp" in topics_over_time_full.columns:
           topics_over_time_full["Timestamp"] = pd.to_datetime(topics_over_time_full["Timestamp"], errors="coerce", utc=True)
   else:
       topics_over_time_full = pd.DataFrame()
       print("topics_over_time_full.csv missing in cached mode; skipping full dynamic topic plot.")


   # Эволюция слов тем (evolution_tuning): пересчитываем только когда нет
   # валидного кэша; в режимах рендера/кэша переиспользуем сохранённый CSV.
   topics_evolution = pd.DataFrame()
   if not args.no_evolution:
       reuse_evolution = topics_evolution_path.exists() and use_cached
       if reuse_evolution:
           topics_evolution = pd.read_csv(topics_evolution_path)
           if "Timestamp" in topics_evolution.columns:
               topics_evolution["Timestamp"] = pd.to_datetime(
                   topics_evolution["Timestamp"], errors="coerce", utc=True
               )
           print(f"--- Loaded cached topic word evolution: {topics_evolution_path.name} ---")
       else:
           topics_evolution = _compute_topics_over_time_evolution(
               topic_model=topic_model,
               model_dir=bertopic_model_dir,
               df_docs=df_topics,
               text_col=args.text_col,
               bin_days=args.sentiment_bin_days,
               nr_bins=args.nr_bins,
           )
           if not topics_evolution.empty:
               topics_evolution.to_csv(topics_evolution_path, index=False)
               print(f"--- Saved topic word evolution: {topics_evolution_path.name} ---")


   # Full-дельта до/после на всём окне анализа с разбиением по отсечке.
   full_sc = scenarios[0]
   comp_full = _build_before_after_table(df_topics, topic_info, full_sc)
   comp_full.to_csv(out_dir / "full_topic_before_after.csv", index=False)
   # Для lineplot берём темы в порядке убывания |delta|
   # (как в fig_delta_share_top_full), но позже ограничим список
   # теми, которые реально присутствуют в topics_over_time_full.
   ordered_topics_by_delta = (
       comp_full[~comp_full["topic_id"].isin([-1, 0])]
       .sort_values("abs_delta_share", ascending=False)
       ["topic_id"]
       .astype(int)
       .tolist()
   )

   if not topics_over_time_full.empty:
       available_topics = set(
           pd.to_numeric(topics_over_time_full["Topic"], errors="coerce")
           .dropna()
           .astype(int)
           .tolist()
       )
       available_topics.difference_update({-1, 0})
       top_topics_from_delta = [t for t in ordered_topics_by_delta if t in available_topics][:5]
       if len(top_topics_from_delta) < 5:
           # Редкий случай: если после пересечения осталось мало тем.
           fallback_topics = sorted(available_topics)
           for t in fallback_topics:
               if t not in top_topics_from_delta:
                   top_topics_from_delta.append(t)
               if len(top_topics_from_delta) >= 5:
                   break

       _plot_dynamic_top_topics(
           topics_over_time=topics_over_time_full,
           topic_lookup=topic_info,
           out_path=out_dir / "fig_topics_over_time_full_top.png",
           top_n=args.top_n_dynamic,
           title="Dynamic topics over time (full analysis period)",
           dpi=args.dpi,
           keep_topics=top_topics_from_delta,
           cutoff_dates=cutoff_dates,
       )
       if not args.no_interactive_topic_plots:
           # При наличии эволюции hover показывает топ-слова темы в каждом бине.
           use_evolution = not topics_evolution.empty
           _plot_dynamic_top_topics_interactive(
               topics_over_time=topics_evolution if use_evolution else topics_over_time_full,
               topic_lookup=topic_info,
               out_path=out_dir / "fig_topics_over_time_full_top_interactive.html",
               top_n=args.top_n_dynamic,
               title=(
                   "Dynamic topics over time (interactive, evolving words per bin)"
                   if use_evolution
                   else "Dynamic topics over time (interactive, full period)"
               ),
               keep_topics=top_topics_from_delta,
               cutoff_dates=cutoff_dates,
           )


   _plot_before_after_delta(
       comparison_df=comp_full,
       out_path=out_dir / "fig_delta_share_top_full.png",
       top_n=args.top_n_delta,
       title="Изменение долей тем (после отсечки - до отсечки), full период",
       dpi=args.dpi,
   )

   if not args.disable_sentiment and "ensemble_label" in df_topics.columns:
       before_mask = (df_topics["date_day"] >= full_sc.before_start) & (df_topics["date_day"] <= full_sc.before_end)
       after_mask = (df_topics["date_day"] >= full_sc.after_start) & (df_topics["date_day"] <= full_sc.after_end)
       sent_before = (
           df_topics[before_mask]["ensemble_label"].value_counts(dropna=False).rename("before_n").to_frame()
       )
       sent_after = (
           df_topics[after_mask]["ensemble_label"].value_counts(dropna=False).rename("after_n").to_frame()
       )
       sent_cmp = sent_before.join(sent_after, how="outer").fillna(0).reset_index(names="sentiment")
       sent_cmp["before_n"] = sent_cmp["before_n"].astype(int)
       sent_cmp["after_n"] = sent_cmp["after_n"].astype(int)
       total_b = max(1, int(sent_cmp["before_n"].sum()))
       total_a = max(1, int(sent_cmp["after_n"].sum()))
       sent_cmp["before_share"] = sent_cmp["before_n"] / total_b
       sent_cmp["after_share"] = sent_cmp["after_n"] / total_a
       sent_cmp["delta_share"] = sent_cmp["after_share"] - sent_cmp["before_share"]
       sent_cmp.to_csv(out_dir / "full_sentiment_before_after.csv", index=False)

   pd.DataFrame(
       [
           {
               "scenario": "full",
               "before_start": full_sc.before_start.date().isoformat(),
               "before_end": full_sc.before_end.date().isoformat(),
               "after_start": full_sc.after_start.date().isoformat(),
               "after_end": full_sc.after_end.date().isoformat(),
               "n_days_after": full_sc.n_days_after,
               "docs_in_scenario": int(len(df_topics)),
               "docs_before": int(((df_topics["date_day"] >= full_sc.before_start) & (df_topics["date_day"] <= full_sc.before_end)).sum()),
               "docs_after": int(((df_topics["date_day"] >= full_sc.after_start) & (df_topics["date_day"] <= full_sc.after_end)).sum()),
           }
       ]
   ).to_csv(out_dir / "scenario_windows_summary.csv", index=False)

   # Оставляем в выходном каталоге только full-артефакты.
   for pat in (
       "fig_v1_*",
       "fig_v2_*",
       "v1_*",
       "v2_*",
   ):
       for old_path in out_dir.glob(pat):
           if old_path.is_file():
               old_path.unlink()


   if (topic_model is not None) and (not args.render_only):
       topic_model.save(
           bertopic_model_dir,
           serialization="safetensors",
           save_ctfidf=True,
           save_embedding_model=False,
       )


   config = {
       "in_jsonl": args.in_jsonl,
       "analysis_start": analysis_start.date().isoformat(),
       "analysis_end": analysis_end.date().isoformat(),
       "cutoff_date": cutoff_date.date().isoformat(),
       "embedding_model_name": args.embedding_model_name,
       "gpu_ids": gpu_ids,
       "nr_bins": args.nr_bins,
       "umap_backend": args.umap_backend,
       "umap_n_neighbors": args.umap_n_neighbors,
       "umap_n_components": args.umap_n_components,
       "umap_min_dist": args.umap_min_dist,
       "umap_metric": args.umap_metric,
       "min_topic_size": args.min_topic_size,
       "min_samples": args.min_samples,
       "cluster_selection_method": args.cluster_selection_method,
       "nr_topics": args.nr_topics,
       "sentiment_models": args.sentiment_models,
       "zero_shot_sentiment_model": args.zero_shot_sentiment_model,
       "zero_shot_labels": args.zero_shot_labels,
       "sentiment_batch_size": args.sentiment_batch_size,
       "sentiment_max_length": args.sentiment_max_length,
       "sentiment_rolling_days": args.sentiment_rolling_days,
       "sentiment_bin_days": args.sentiment_bin_days,
       "disable_sentiment": args.disable_sentiment,
       "recompute_sentiment_only": args.recompute_sentiment_only,
       "recompute_topics_only": args.recompute_topics_only,
       "filter_query_stop_terms": FILTER_QUERY_STOP_TERMS,
       "no_interactive_topic_plots": args.no_interactive_topic_plots,
       "no_evolution": args.no_evolution,
   }
   (out_dir / "config.json").write_text(
       json.dumps(config, ensure_ascii=False, indent=2),
       encoding="utf-8",
   )


   print("Saved dynamic BERTopic outputs to:", out_dir)




if __name__ == "__main__":
   os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
   main()
