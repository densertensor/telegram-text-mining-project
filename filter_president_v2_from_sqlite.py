#!/usr/bin/env python3
"""Президентский фильтр v2 — по итогам многосторонней LLM-валидации (2026-06-11).

Изменения к v1 (filter_putin_posts_from_sqlite.py):
1. Футер-чистка и исключение рейтинг-листов до матчинга (как в части «исполнители»).
2. Страновой гард расширен: существительные CLDR (окно ±2) + иностранные
   прилагательные (±3) + фамилии иностранных лидеров (±3) от «президент*».
3. Whitelist-override: пост НЕ исключается гардом, если в нём есть путин*
   (не только в хэштеге) или «президент* росси*/рф/российск*» — возвращает
   ~8.5k дипломатических mixed-постов, выброшенных v1.
4. верховн*: к гарду «суд» добавлены рад*/лидер*/комиссар*/представит* (±1) и нато (±2).
5. Структурные исключения «президент»: «офис президента», «президентск*
   (дворец|бригад|резиденц|комплекс)», «президент (уефа|фифа|клуба|компании|федерации)».
6. Гард «путин только в хэштеге».
7. Новые алиасы: ввп (минус экономический контекст), «перв* лиц*» (минус «от первого
   лица»/FPV, плюс контекст власти), нацлидер (минус Кадыров/Зеленский), главковерх*,
   putin (латиница). «Царь» НЕ добавлен (94–96% шума по аудиту).
8. Выходной флаг _mixed (Путин + иностранный лидер в одном посте) и _triggers.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from babel import Locale
from pymorphy3 import MorphAnalyzer
from razdel import tokenize as razdel_tokenize

MORPH = MorphAnalyzer(lang="ru")


def norm_text(s: str) -> str:
    return s.lower().replace("ё", "е")


def tokenize(s: str) -> list[str]:
    out = []
    for t in razdel_tokenize(norm_text(s)):
        if any(ch.isalnum() for ch in t.text):
            out.append(t.text)
    return out


@lru_cache(maxsize=200_000)
def lemma_ru_token(token: str) -> str:
    if not token:
        return token
    parsed = MORPH.parse(token)
    if not parsed:
        return token
    return norm_text(parsed[0].normal_form)


# --- Страновой индекс CLDR (как в v1) -------------------------------------

def load_country_index_ru():
    ru = Locale.parse("ru")
    stop_words = {
        "и", "республика", "федерация", "королевство", "штаты", "острова",
        "остров", "демократическая", "народная", "соединенные",
        "центральноафриканская",
    }
    exact: dict[str, set] = defaultdict(set)
    lemmas: dict[str, set] = defaultdict(set)
    for code, name in ru.territories.items():
        if not (len(code) == 2 and code.isalpha()) or code.upper() == "RU":
            continue
        country = str(name)
        if "неизвестн" in norm_text(country):
            continue
        toks = [t for t in tokenize(country) if len(t) >= 4 and t not in stop_words]
        if not toks:
            continue
        core = toks[-1]
        if core == "того":
            continue
        exact[core].add(country)
        lemmas[lemma_ru_token(core)].add(country)
    for tok in ("россия", "российская", "российской", "рф", "русский", "русская"):
        exact.pop(tok, None)
        lemmas.pop(lemma_ru_token(tok), None)
    for alias, country in (("сша", "США"), ("америка", "США"), ("британия", "Великобритания")):
        exact[alias].add(country)
        lemmas[lemma_ru_token(alias)].add(country)
    return dict(exact), dict(lemmas)


EXACT_COUNTRY, LEMMA_COUNTRY = load_country_index_ru()

FOREIGN_ADJ_PREFIXES = (
    "украинск", "киевск", "американск", "французск", "немецк", "британск",
    "польск", "болгарск", "израильск", "иранск", "турецк", "казахстанск",
    "белорусск", "армянск", "азербайджанск", "молдавск", "грузинск",
    "эстонск", "латвийск", "литовск", "сирийск", "венесуэльск", "финск",
    "чешск", "словацк", "румынск", "венгерск", "сербск", "южнокорейск",
    "севернокорейск", "китайск", "индийск", "бразильск", "аргентинск",
)
FOREIGN_LEADER_PREFIXES = (
    "зеленск", "трамп", "байден", "макрон", "эрдоган", "орбан", "вучич",
    "алиев", "пашинян", "лукашенк", "асад", "мадуро", "дуда", "навроцк",
    "туск", "фицо", "науседа", "санду", "токаев", "рахмон", "мирзиеев",
    "болсонару", "пезешкиан", "рамафос", "кеннеди", "ющенко", "порошенк",
    "янукович", "обам", "клинтон", "буш", "никсон", "рейган", "олланд",
    "саркози", "шольц", "мерц", "стармер", "милей", "штайнмаер",
)
PREZIDENT_ORG_NEXT = ("уефа", "фифа", "клуба", "компании", "федерации", "ассоциации", "академии")
PREZIDENT_ADJ_NEXT = ("дворец", "дворц", "бригад", "резиденц", "комплекс")


def country_near(tokens, idx, window):
    left, right = max(0, idx - window), min(len(tokens), idx + window + 1)
    for w in tokens[left:right]:
        if w == "того":
            continue
        cs = EXACT_COUNTRY.get(w)
        if cs and len(cs) == 1:
            return next(iter(cs))
        cs = LEMMA_COUNTRY.get(lemma_ru_token(w))
        if cs and len(cs) == 1:
            return next(iter(cs))
    return None


def window_has_prefix(tokens, idx, radius, prefixes):
    left, right = max(0, idx - radius), min(len(tokens), idx + radius + 1)
    return any(any(t.startswith(p) for p in prefixes) for t in tokens[left:right])


# --- Футеры и рейтинг-листы (как в части «исполнители») --------------------

FOOTER_LINE_RE = re.compile(
    r"(подпис(?:аться|ывайся|ывайтесь|ка|чик)|t\.me/|max\.ru|"
    r"кана[лм]\w*\s+в\s+max|мы\s+в\s+max|соловьев\s+в\s+max|"
    r"\bбуст\w*|\bboost|вступай\w*|наш\s+канал|прислать\s+новост|"
    r"обратная\s+связь|поддержать\s+(?:нас|канал)|резервн\w+\s+канал)",
    re.IGNORECASE,
)
TOP_LIST_RE = re.compile(r"топ\s*[-–—]?\s*\d{2,3}", re.IGNORECASE)


def strip_footer_lines(text: str) -> tuple[str, int]:
    # Второй проход: голая короткая строка (<=2 токенов) рядом с подписной —
    # кросс-промо названия канала («РОСКОМНАДЗОР\n\nПодписаться на канал»),
    # а не упоминание персоны; тоже футер. Подпись источника в конце обычного
    # текста (без подписной строки рядом) сохраняется.
    lines = text.split("\n")
    flagged = [
        len(line) <= 200 and bool(FOOTER_LINE_RE.search(norm_text(line)))
        for line in lines
    ]
    nonempty_idx = [i for i, line in enumerate(lines) if line.strip()]
    for pos, i in enumerate(nonempty_idx):
        if flagged[i] or len(lines[i]) > 60:
            continue
        toks = norm_text(lines[i]).split()
        if not toks or len(toks) > 2:
            continue
        if pos + 1 < len(nonempty_idx) and flagged[nonempty_idx[pos + 1]]:
            flagged[i] = True
    kept = [line for line, f in zip(lines, flagged) if not f]
    return "\n".join(kept), sum(flagged)


def is_rating_list(norm: str, raw: str) -> bool:
    if TOP_LIST_RE.search(norm) and "канал" in norm:
        return True
    if raw.count("@") >= 8 and "канал" in norm and ("подборк" in norm or "топ" in norm or "рейтинг" in norm):
        return True
    return False


# --- Матчер v2 --------------------------------------------------------------

VERKHOVN_BAD_NEXT = ("суд", "рад", "лидер", "комиссар", "представит")
VVP_ECON_PREFIXES = ("рост", "росл", "процент", "трлн", "млрд", "доллар", "юан",
                     "эконом", "инфляц", "дефицит", "номинал", "паритет", "душ")
PL_BAD = ("fpv", "фпв", "дрон")
PL_CTX = ("власт", "государств", "стран", "кремл", "решени", "доклад", "верховн", "главнокоманд")
NATSLIDER_BAD = ("кадыров", "зеленск", "чечн", "украин")


def match_president_v2(tokens: list[str], norm_full: str) -> tuple[bool, dict, list[tuple[str, str]]]:
    """Возвращает (include, info, exclusions[(trigger, guard)])."""
    excl: list[tuple[str, str]] = []
    triggers: set[str] = set()
    foreign_seen = False

    # путин* (+ гард «только в хэштеге»)
    putin_hit = any(t.startswith("путин") for t in tokens)
    if putin_hit:
        n_all = norm_full.count("путин")
        n_hash = norm_full.count("#путин")
        if n_all > 0 and n_all == n_hash:
            putin_hit = False
            excl.append(("putin", "hashtag_only"))
        else:
            triggers.add("putin")

    # президент* росси* / рф — компонент override
    pres_rossii = False
    for i, t in enumerate(tokens):
        if t.startswith("президент"):
            for j in range(i + 1, min(len(tokens), i + 3)):
                if tokens[j].startswith(("росси", "рф", "российск")):
                    pres_rossii = True
                    break
    override = putin_hit or pres_rossii

    # президент* с гардами
    pres_clean = False
    for i, t in enumerate(tokens):
        if not t.startswith("президент"):
            continue
        guard = None
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        prv = tokens[i - 1] if i > 0 else ""
        if prv.startswith("офис"):
            guard = "office_of_president"
        elif t.startswith("президентск") and any(nxt.startswith(p) for p in PREZIDENT_ADJ_NEXT):
            guard = "presidential_object"
        elif nxt in PREZIDENT_ORG_NEXT:
            guard = "president_of_org"
        else:
            c = country_near(tokens, i, 2)
            if c:
                guard = f"country:{c}"
            elif window_has_prefix(tokens, i, 3, FOREIGN_ADJ_PREFIXES):
                guard = "foreign_adj"
            elif window_has_prefix(tokens, i, 3, FOREIGN_LEADER_PREFIXES):
                guard = "foreign_leader"
        if guard is None:
            pres_clean = True
        else:
            foreign_seen = True
            if override:
                pres_clean = True  # mixed-пост, сохраняем
            else:
                excl.append(("prezident", guard))
    if pres_clean:
        triggers.add("prezident")

    # верховн* с расширенными гардами
    for i, t in enumerate(tokens):
        if not t.startswith("верховн"):
            continue
        left, right = max(0, i - 1), min(len(tokens), i + 2)
        neigh = tokens[left:right]
        if any(any(x.startswith(p) for p in VERKHOVN_BAD_NEXT) for x in neigh):
            excl.append(("verkhovn", "rada_lider_komissar_sud"))
            continue
        if window_has_prefix(tokens, i, 2, ("нато",)):
            excl.append(("verkhovn", "nato"))
            continue
        triggers.add("verkhovn")
        break

    # владимир /1 владимирович
    for i, t in enumerate(tokens):
        if t.startswith("владимир"):
            for j in range(i + 1, min(len(tokens), i + 3)):
                if tokens[j].startswith("владимирович"):
                    triggers.add("vladimir_vladimirovich")
                    break

    # --- алиасы v2 ---
    for i, t in enumerate(tokens):
        if t == "ввп":
            if window_has_prefix(tokens, i, 6, VVP_ECON_PREFIXES):
                excl.append(("vvp", "econ_context"))
            else:
                triggers.add("vvp")
        if t in ("первое", "первого", "первому") and i + 1 < len(tokens) and tokens[i + 1].startswith("лиц"):
            if i > 0 and tokens[i - 1] == "от":
                excl.append(("pervoe_litso", "ot_pervogo_litsa"))
            elif window_has_prefix(tokens, i, 5, PL_BAD):
                excl.append(("pervoe_litso", "fpv_context"))
            elif window_has_prefix(tokens, i, 10, PL_CTX):
                triggers.add("pervoe_litso")
            else:
                excl.append(("pervoe_litso", "no_power_context"))
        if t.startswith("нацлидер") or (
            t.startswith("национальн") and any(x.startswith("лидер") for x in tokens[i + 1:i + 3])
        ):
            if window_has_prefix(tokens, i, 5, NATSLIDER_BAD):
                excl.append(("natslider", "other_person"))
            else:
                triggers.add("natslider")
        if t.startswith("главковерх"):
            triggers.add("glavkoverh")
        if t.startswith("putin"):
            triggers.add("putin_latin")

    info = {
        "triggers": sorted(triggers),
        "mixed": bool(triggers) and (
            foreign_seen
            or any(any(t.startswith(p) for p in FOREIGN_LEADER_PREFIXES) for t in tokens)
        ),
    }
    return bool(triggers), info, excl


PRESCREEN = re.compile(
    r"путин|президент|верховн|владимирович|\bввп\b|перво[ег]\w*\s+лиц|нацлидер|"
    r"национальн\w*\s+лидер|главковерх|putin"
)


def views_to_int(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.replace(" ", "").strip()
        try:
            return int(float(s)) if s else None
        except ValueError:
            return None
    return None


def snippet(text: str, limit: int = 180) -> str:
    s = " ".join(text.split())
    return s[: limit - 3] + "..." if len(s) > limit else s


def main() -> None:
    p = argparse.ArgumentParser(description="Президентский фильтр v2.")
    p.add_argument("--db-path", default="patriot_channels_posts_20260423_233414.sqlite")
    p.add_argument("--out-jsonl", default="outputs/president_putin_selection_v2.jsonl")
    p.add_argument("--reports-dir", default="outputs")
    p.add_argument("--cutoffs", default="2026-01-16")
    p.add_argument("--log-every", type=int, default=200000)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cutoffs = [date.fromisoformat(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    conn = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    sql = "SELECT id, channel_username, payload_json FROM posts_raw"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    cur.execute(sql)

    excl_counter: Counter = Counter()
    excl_examples: dict = defaultdict(list)
    trig_counter: Counter = Counter()
    date_counts: Counter = Counter()
    mixed_count = 0
    rating_skipped = 0
    footer_removed = 0
    written = 0
    total = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as out:
        for src_id, ch_user, payload_json in cur:
            total += 1
            if total % args.log_every == 0:
                print(f"[{time.time()-t0:8.1f}s] processed={total:,} written={written:,}", flush=True)
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                continue
            text = payload.get("text") or ""
            if not text:
                continue
            norm_full = norm_text(text)
            if not PRESCREEN.search(norm_full):
                continue

            clean_text, removed = strip_footer_lines(text)
            footer_removed += removed
            norm = norm_text(clean_text)
            if not PRESCREEN.search(norm):
                excl_counter[("_post", "footer_only_match")] += 1
                continue
            if is_rating_list(norm, clean_text):
                rating_skipped += 1
                excl_counter[("_post", "rating_list")] += 1
                if len(excl_examples[("_post", "rating_list")]) < 8:
                    excl_examples[("_post", "rating_list")].append(snippet(text))
                continue

            tokens = tokenize(clean_text)
            include, info, exclusions = match_president_v2(tokens, norm)
            for trig, guard in exclusions:
                excl_counter[(trig, guard)] += 1
                if len(excl_examples[(trig, guard)]) < 8:
                    excl_examples[(trig, guard)].append(snippet(clean_text))
            if not include:
                continue

            v = views_to_int(payload.get("views"))
            if v is None or v < 1:
                continue

            rec = dict(payload)
            rec["_triggers"] = info["triggers"]
            rec["_mixed"] = info["mixed"]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if info["mixed"]:
                mixed_count += 1
            for t in info["triggers"]:
                trig_counter[t] += 1
            ds = payload.get("date")
            if ds:
                try:
                    date_counts[datetime.fromisoformat(ds).date()] += 1
                except ValueError:
                    pass

    conn.close()

    print("=== RESULT v2 ===")
    print(f"total_scanned: {total}")
    print(f"written (views>=1): {written}  (v1 было 169 702 в окне анализа)")
    print(f"mixed: {mixed_count}")
    print(f"rating_lists_skipped: {rating_skipped}; footer_lines_removed: {footer_removed}")
    print("triggers:", trig_counter.most_common())
    print("top exclusions:", excl_counter.most_common(20))

    with (reports_dir / "president_v2_exclusions_report.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trigger", "guard", "count", "examples"])
        for (trig, guard), cnt in sorted(excl_counter.items(), key=lambda x: -x[1]):
            w.writerow([trig, guard, cnt, json.dumps(excl_examples.get((trig, guard), [])[:8], ensure_ascii=False)])

    summary = {
        "total_scanned": total,
        "written": written,
        "mixed": mixed_count,
        "rating_lists_skipped": rating_skipped,
        "footer_lines_removed": footer_removed,
        "trigger_counts": dict(trig_counter),
        "exclusion_counts": {f"{t}|{g}": c for (t, g), c in excl_counter.items()},
    }
    (reports_dir / "president_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.limit and date_counts:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dates = sorted(date_counts)
        with (reports_dir / "president_v2_by_date.csv").open("w", encoding="utf-8") as f:
            f.write("date,posts_count\n")
            for d in dates:
                f.write(f"{d.isoformat()},{date_counts[d]}\n")
        plt.figure(figsize=(14, 6))
        x = [d.isoformat() for d in dates]
        plt.plot(x, [date_counts[d] for d in dates], linewidth=1.2)
        for c in cutoffs:
            plt.axvline(c.isoformat(), linestyle="--", color="black", linewidth=1.4)
        plt.title("Президентская селекция v2 по датам")
        plt.grid(True, alpha=0.3)
        step = max(1, len(x) // 15)
        plt.xticks(x[::step], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(reports_dir / "president_v2_by_date.png", dpi=180)
        print("plot/csv written")


if __name__ == "__main__":
    main()
