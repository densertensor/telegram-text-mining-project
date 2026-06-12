#!/usr/bin/env python3
"""Подготовка данных для ревизии президентского запроса.

Часть 1 (precision): стратифицированные выборки из president_putin_selection
по триггерам включения (путин / президент / верховн / владимир+владимирович)
и отдельная страта «президент рядом с иностранным маркером (прилагательное/
фамилия лидера)» — кандидат на утечку шума мимо странового гарда (окно ±1,
только существительные-страны).

Часть 2 (recall): скан posts_raw на алиасы Путина, НЕ покрытые запросом
(ввп, царь, главковерх, нацлидер, «первое лицо», putin латиницей,
«хозяин кремля», пыня) — счётчики + сохранение постов-кандидатов.

Выход: outputs/president_audit/{precision_samples.jsonl, recall_candidates.jsonl,
recall_counts.csv, prep_summary.json}
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from razdel import tokenize as razdel_tokenize

SELECTION_JSONL = "outputs/president_putin_selection_20260423_233414.jsonl"
DB_PATH = "patriot_channels_posts_20260423_233414.sqlite"
OUT_DIR = Path("outputs/president_audit")
CUTOFF = "2026-01-16"
START = "2022-02-24"
SAMPLE_PER_STRATUM = 60
SEED = 42


def norm_text(s: str) -> str:
    return s.lower().replace("ё", "е")


def tokenize(s: str) -> list[str]:
    out = []
    for t in razdel_tokenize(norm_text(s)):
        if any(ch.isalnum() for ch in t.text):
            out.append(t.text)
    return out


FOREIGN_ADJ_PREFIXES = (
    "украинск", "киевск", "американск", "французск", "немецк", "британск",
    "польск", "болгарск", "израильск", "иранск", "турецк", "казахстанск",
    "белорусск", "армянск", "азербайджанск", "молдавск", "грузинск",
    "эстонск", "латвийск", "литовск", "сирийск", "венесуэльск", "финск",
    "чешск", "словацк", "румынск", "венгерск", "сербск", "бывш",
)
FOREIGN_LEADER_PREFIXES = (
    "зеленск", "байден", "трамп", "макрон", "эрдоган", "алиев", "пашинян",
    "лукашенк", "асад", "мадуро", "дуда", "науседа", "санду", "вучич",
    "орбан", "фицо", "токаев", "рахмон", "мирзиеев", "си", "ын",
)


def detect_triggers(tokens: list[str]) -> dict:
    trig = {
        "putin": False,
        "prezident": False,
        "verkhovn": False,
        "vladimir_vladimirovich": False,
        "verkhovn_rada": False,
        "prezident_foreign": False,
    }
    for i, t in enumerate(tokens):
        if t.startswith("путин"):
            trig["putin"] = True
        if t.startswith("президент"):
            trig["prezident"] = True
            left = max(0, i - 3)
            right = min(len(tokens), i + 4)
            for w in tokens[left:right]:
                if any(w.startswith(p) for p in FOREIGN_ADJ_PREFIXES) or any(
                    w.startswith(p) for p in FOREIGN_LEADER_PREFIXES
                ):
                    trig["prezident_foreign"] = True
                    break
        if t.startswith("верховн"):
            left = max(0, i - 1)
            right = min(len(tokens), i + 2)
            if not any(x.startswith("суд") for x in tokens[left:right]):
                trig["verkhovn"] = True
            if any(x.startswith("рад") for x in tokens[left:right]):
                trig["verkhovn_rada"] = True
        if t.startswith("владимир"):
            for j in range(i + 1, min(len(tokens), i + 3)):
                if tokens[j].startswith("владимирович"):
                    trig["vladimir_vladimirovich"] = True
    return trig


def include_query(tokens: list[str]) -> bool:
    for i, t in enumerate(tokens):
        if t.startswith("путин") or t.startswith("президент"):
            return True
        if t.startswith("верховн"):
            left = max(0, i - 1)
            right = min(len(tokens), i + 2)
            if not any(x.startswith("суд") for x in tokens[left:right]):
                return True
        if t.startswith("владимир"):
            for j in range(i + 1, min(len(tokens), i + 3)):
                if tokens[j].startswith("владимирович"):
                    return True
    return False


class Reservoir:
    def __init__(self, k: int, rng: random.Random):
        self.k = k
        self.rng = rng
        self.items: list = []
        self.seen = 0

    def add(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.k:
                self.items[j] = item


def period_of(date_s: str) -> str | None:
    d = (date_s or "")[:10]
    if not d or d < START:
        return None
    return "before" if d < CUTOFF else "after"


def run_precision() -> dict:
    rng = random.Random(SEED)
    strata: dict[tuple[str, str], Reservoir] = defaultdict(lambda: Reservoir(SAMPLE_PER_STRATUM, rng))
    trigger_counts: Counter = Counter()
    total = 0
    t0 = time.time()
    with open(SELECTION_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            period = period_of(r.get("date"))
            if period is None:
                continue
            total += 1
            if total % 50000 == 0:
                print(f"[precision {time.time()-t0:7.1f}s] processed={total:,}", flush=True)
            text = r.get("text") or ""
            tokens = tokenize(text)
            trig = detect_triggers(tokens)
            item = {
                "date": r.get("date"),
                "channel": r.get("channel_username"),
                "text": " ".join(text.split())[:1500],
                "triggers": {k: v for k, v in trig.items() if v},
            }
            for key in ("putin", "prezident", "verkhovn", "vladimir_vladimirovich"):
                if trig[key]:
                    trigger_counts[(key, period)] += 1
                    strata[(key, period)].add(item)
            # сфокусированные страты-гипотезы
            if trig["prezident_foreign"]:
                trigger_counts[("prezident_foreign", period)] += 1
                strata[("prezident_foreign", period)].add(item)
            if trig["verkhovn_rada"]:
                trigger_counts[("verkhovn_rada", period)] += 1
                strata[("verkhovn_rada", period)].add(item)
            # «чистый верховный»: триггер только верховн, без путин/президент
            if trig["verkhovn"] and not trig["putin"] and not trig["prezident"]:
                trigger_counts[("verkhovn_only", period)] += 1
                strata[("verkhovn_only", period)].add(item)

    out_path = OUT_DIR / "precision_samples.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for (key, period), res in sorted(strata.items()):
            for item in res.items:
                rec = dict(item)
                rec["stratum"] = key
                rec["period"] = period
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    counts = {f"{k}|{p}": c for (k, p), c in trigger_counts.items()}
    print("precision strata counts:", json.dumps(counts, ensure_ascii=False, indent=1))
    return {"total_in_window": total, "trigger_counts": counts, "samples": str(out_path)}


ALIAS_CHECKS = {
    "vvp": lambda toks: "ввп" in toks,
    "tsar": lambda toks: any(t in ("царь", "царя", "царю", "царем", "царе") for t in toks),
    "glavkoverh": lambda toks: any(t.startswith("главковерх") for t in toks),
    "natslider": lambda toks: any(t.startswith("нацлидер") for t in toks)
    or any(
        t.startswith("национальн") and any(x.startswith("лидер") for x in toks[max(0, i - 1):i + 3])
        for i, t in enumerate(toks)
    ),
    "pervoe_litso": lambda toks: any(
        t in ("первое", "первого", "первому") and i + 1 < len(toks) and toks[i + 1].startswith("лиц")
        for i, t in enumerate(toks)
    ),
    "putin_latin": lambda toks: any(t.startswith("putin") for t in toks),
    "hozyain_kremlya": lambda toks: any(
        t.startswith("хозяин") and any(x.startswith("кремл") for x in toks[i + 1:i + 4])
        for i, t in enumerate(toks)
    ),
    "pynya": lambda toks: "пыня" in toks,
}
ALIAS_PRESCREEN = re.compile(
    r"ввп|цар[ьяюее]|главковерх|нацлидер|национальн\w*\s+лидер|перво[ег]\w*\s+лиц|putin|хозяин\w*\s+кремл|пыня"
)
RECALL_KEEP_PER_ALIAS = 4000  # верхняя граница сохранённых постов-кандидатов на алиас


def run_recall() -> dict:
    rng = random.Random(SEED + 1)
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT channel_username, post_date, payload_json FROM posts_raw")
    totals: Counter = Counter()        # (alias, period) — все вхождения
    uncovered: Counter = Counter()     # (alias, period) — вхождения вне запроса
    keepers: dict[str, Reservoir] = defaultdict(lambda: Reservoir(RECALL_KEEP_PER_ALIAS, rng))
    total = 0
    t0 = time.time()
    for ch, date_s, payload_json in cur:
        total += 1
        if total % 500000 == 0:
            print(f"[recall {time.time()-t0:7.1f}s] processed={total:,}", flush=True)
        period = period_of(date_s or "")
        if period is None:
            continue
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            continue
        text = payload.get("text") or ""
        if not text:
            continue
        norm = norm_text(text)
        if not ALIAS_PRESCREEN.search(norm):
            continue
        toks = tokenize(text)
        hits = [a for a, check in ALIAS_CHECKS.items() if check(toks)]
        if not hits:
            continue
        covered = include_query(toks)
        for a in hits:
            totals[(a, period)] += 1
            if not covered:
                uncovered[(a, period)] += 1
                keepers[a].add({
                    "alias": a,
                    "period": period,
                    "date": date_s,
                    "channel": ch,
                    "text": " ".join(text.split())[:1500],
                })
    conn.close()

    out_path = OUT_DIR / "recall_candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for a, res in sorted(keepers.items()):
            for item in res.items:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (OUT_DIR / "recall_counts.csv").open("w", encoding="utf-8") as f:
        f.write("alias,period,total_mentions,uncovered_by_query\n")
        for (a, p) in sorted(totals):
            f.write(f"{a},{p},{totals[(a, p)]},{uncovered.get((a, p), 0)}\n")
    print("recall totals:", dict(totals))
    print("recall uncovered:", dict(uncovered))
    return {
        "totals": {f"{a}|{p}": c for (a, p), c in totals.items()},
        "uncovered": {f"{a}|{p}": c for (a, p), c in uncovered.items()},
        "candidates": str(out_path),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"precision": run_precision(), "recall": run_recall()}
    (OUT_DIR / "prep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
