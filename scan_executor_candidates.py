#!/usr/bin/env python3
"""Разведочный скан posts_raw под запрос по ведомствам-исполнителям.

Один полный проход по 4.4M постов:
- широкий набор паттернов-кандидатов (акторы: Роскомнадзор, РКН, Минцифры,
  Шадаев и пр.; контекст: блокировки/замедления мессенджеров и платформ);
- посты, совпавшие хотя бы с одним актор-паттерном или ключевым контекстным
  ко-вхождением, пишутся в рабочую SQLite-БД executor_candidates.sqlite
  с булевыми флагами по каждому паттерну;
- для ВСЕХ паттернов (включая слишком широкие, чьи тексты не сохраняем)
  считаются помесячные счётчики -> CSV.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

SRC_DB = "patriot_channels_posts_20260423_233414.sqlite"
OUT_DB = "outputs/executor_candidates.sqlite"
OUT_MONTHLY_CSV = "outputs/executor_terms_monthly_counts.csv"
OUT_TOTALS_CSV = "outputs/executor_terms_totals.csv"


def norm_text(s: str) -> str:
    return s.lower().replace("ё", "е")


# Актор-паттерны: совпадение => пост сохраняется с текстом.
ACTOR_PATTERNS: dict[str, re.Pattern] = {
    # Роскомнадзор и его инфраструктура
    "roskomnadzor": re.compile(r"роскомнадзор"),
    "rkn_abbr": re.compile(r"\bркн\b"),
    "grchc": re.compile(r"\bгрчц\b"),
    "radiochastotny_centr": re.compile(r"радиочастотн\w*\s+центр"),
    "cmu_ssop": re.compile(r"\bссоп\b"),
    "tspu": re.compile(r"\bтспу\b"),
    "lipov": re.compile(r"\bлипов(?:а|у|ым|е|ой)?\b"),
    "zharov": re.compile(r"\bжаров\w*\b"),
    # Минцифры и руководство
    "mincifry": re.compile(r"минцифр"),
    "minsvyaz": re.compile(r"\bминсвяз"),
    "ministerstvo_cifrovogo": re.compile(r"министерств\w*\s+цифров"),
    "shadaev": re.compile(r"шадаев"),
    # Законодатели / комментаторы блокировок
    "klishas": re.compile(r"клишас"),
    "gorelkin": re.compile(r"горелкин"),
    # Депутат Свинцов (ЛДПР, комитет по информполитике); регэксп отсекает
    # большинство форм прилагательного "свинцовый", но "свинцовым" омонимично.
    "svintsov": re.compile(r"\bсвинцов(?:а|у|ым|е|ой)?\b"),
    "boyarsky": re.compile(r"боярск"),
    "mizulina": re.compile(r"мизулин"),
    # Средний уровень (опциональный объект поддержки)
    "mishustin": re.compile(r"мишустин"),
    "kabmin": re.compile(r"\bкабмин"),
    # Прочие причастные ведомства
    "fas_svyaz": re.compile(r"\bфас\b"),
    "prokuratura_gen": re.compile(r"генпрокуратур"),
}

# «Политическое руководство» помимо Путина (верх лестницы Норрис):
# совпадение => пост сохраняется с текстом (отдельный сёрч от исполнителей).
LEADERSHIP_PATTERNS: dict[str, re.Pattern] = {
    "kreml": re.compile(r"кремл"),
    "adm_prezidenta": re.compile(r"администраци\w*\s+президента"),
    "ap_abbr": re.compile(r"\bап\b"),
    "vaino": re.compile(r"вайно"),
    "kirienko": re.compile(r"кириенко"),
    "gromov": re.compile(r"\bгромов\w*\b"),
    "peskov": re.compile(r"\bпесков(?:а|у|ым|е)?\b"),
    "sovbez": re.compile(r"совбез"),
    "sovet_bezopasnosti": re.compile(r"совет\w*\s+безопасности"),
    "patrushev": re.compile(r"патрушев"),
    "shoigu": re.compile(r"шойгу"),
    "medvedev": re.compile(r"медведев"),
    "volodin": re.compile(r"володин"),
    "matvienko": re.compile(r"матвиенко"),
    "garant": re.compile(r"\bгарант(?:а|у|ом|е)?\b"),
}

# Контекстные паттерны (для флагов и счётчиков).
CONTEXT_PATTERNS: dict[str, re.Pattern] = {
    "telegram_lat": re.compile(r"telegram"),
    "telegram_cyr": re.compile(r"телеграм"),
    "messenger": re.compile(r"мессенджер"),
    "whatsapp": re.compile(r"whatsapp|вотсап|ватсап"),
    "youtube": re.compile(r"youtube|ютуб|ютьюб"),
    "max_messenger": re.compile(r"мессенджер\w*\s+max|\bmax\b"),
    "block": re.compile(r"блокир|заблокир"),
    "slowdown": re.compile(r"замедл"),
    "restrict": re.compile(r"ограничен"),
    "vpn": re.compile(r"\bvpn\b|\bвпн\b"),
    "runet": re.compile(r"рунет"),
    "white_list": re.compile(r"бел\w+\s+спис"),
    "shutdown": re.compile(r"шатдаун|отключени\w*\s+(?:мобильного\s+)?интернет"),
    "cenzura": re.compile(r"цензур"),
}

PLATFORM_KEYS = ["telegram_lat", "telegram_cyr", "messenger", "whatsapp", "youtube", "vpn", "runet"]
RESTRICT_KEYS = ["block", "slowdown", "restrict", "shutdown", "cenzura", "white_list"]

ALL_KEYS = (
    list(ACTOR_PATTERNS)
    + list(LEADERSHIP_PATTERNS)
    + list(CONTEXT_PATTERNS)
    + ["ctx_platform_x_restrict"]
)


def main() -> None:
    out_db_path = Path(OUT_DB)
    out_db_path.parent.mkdir(parents=True, exist_ok=True)
    if out_db_path.exists():
        out_db_path.unlink()

    out = sqlite3.connect(str(out_db_path))
    flag_cols = ",\n".join(f"            {k} INTEGER NOT NULL DEFAULT 0" for k in ALL_KEYS)
    # journal_mode=DELETE: вывод пишется на сетевую ФС, WAL на NFS ненадёжен.
    out.executescript(
        f"""
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = OFF;
        CREATE TABLE candidates (
            src_id INTEGER PRIMARY KEY,
            channel_username TEXT,
            post_id INTEGER,
            post_date TEXT,
            views INTEGER,
            text TEXT,
            post_url TEXT,
            matched_actors TEXT,
            matched_leaders TEXT,
{flag_cols}
        );
        """
    )
    insert_cols = [
        "src_id", "channel_username", "post_id", "post_date", "views",
        "text", "post_url", "matched_actors", "matched_leaders",
    ] + ALL_KEYS
    insert_sql = (
        f"INSERT INTO candidates ({', '.join(insert_cols)}) "
        f"VALUES ({', '.join('?' for _ in insert_cols)})"
    )

    monthly: Counter = Counter()  # (term, YYYY-MM) -> n
    totals: Counter = Counter()

    src = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
    cur = src.cursor()
    cur.execute("SELECT id, channel_username, post_id, post_date, payload_json FROM posts_raw")

    t0 = time.time()
    total = 0
    saved = 0
    batch: list[tuple] = []

    for src_id, ch_user, post_id, post_date, payload_json in cur:
        total += 1
        if total % 500000 == 0:
            print(f"[{time.time()-t0:7.1f}s] processed={total:,} saved={saved:,}", flush=True)
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            continue
        text = payload.get("text") or ""
        if not text:
            continue
        norm = norm_text(text)

        flags: dict[str, int] = {}
        matched_actor = False
        for key, pat in ACTOR_PATTERNS.items():
            hit = 1 if pat.search(norm) else 0
            flags[key] = hit
            if hit:
                matched_actor = True
        matched_leader = False
        for key, pat in LEADERSHIP_PATTERNS.items():
            hit = 1 if pat.search(norm) else 0
            flags[key] = hit
            if hit:
                matched_leader = True
        for key, pat in CONTEXT_PATTERNS.items():
            flags[key] = 1 if pat.search(norm) else 0

        platform_hit = any(flags[k] for k in PLATFORM_KEYS)
        restrict_hit = any(flags[k] for k in RESTRICT_KEYS)
        flags["ctx_platform_x_restrict"] = 1 if (platform_hit and restrict_hit) else 0

        month = (post_date or "")[:7]
        for key, val in flags.items():
            if val:
                monthly[(key, month)] += 1
                totals[key] += 1

        if matched_actor or matched_leader or flags["ctx_platform_x_restrict"]:
            saved += 1
            views = payload.get("views")
            try:
                views = int(float(views)) if views is not None else None
            except (TypeError, ValueError):
                views = None
            matched_actors = ",".join(k for k in ACTOR_PATTERNS if flags[k])
            matched_leaders = ",".join(k for k in LEADERSHIP_PATTERNS if flags[k])
            post_url = payload.get("post_url")
            batch.append(
                (
                    src_id, ch_user, post_id, post_date, views,
                    text, post_url, matched_actors, matched_leaders,
                )
                + tuple(flags[k] for k in ALL_KEYS)
            )
            if len(batch) >= 2000:
                out.executemany(insert_sql, batch)
                out.commit()
                batch = []

    if batch:
        out.executemany(insert_sql, batch)
        out.commit()

    out.execute("CREATE INDEX idx_cand_date ON candidates(post_date)")
    out.execute("CREATE INDEX idx_cand_actors ON candidates(matched_actors)")
    out.execute("CREATE INDEX idx_cand_leaders ON candidates(matched_leaders)")
    out.commit()

    with open(OUT_MONTHLY_CSV, "w", encoding="utf-8") as f:
        f.write("term,month,posts_n\n")
        for (term, month), n in sorted(monthly.items()):
            f.write(f"{term},{month},{n}\n")
    with open(OUT_TOTALS_CSV, "w", encoding="utf-8") as f:
        f.write("term,posts_n\n")
        for term, n in totals.most_common():
            f.write(f"{term},{n}\n")

    src.close()
    out.close()
    print(f"DONE in {time.time()-t0:.1f}s: processed={total:,} saved={saved:,}")
    print(f"out_db={OUT_DB}")
    print(f"monthly_csv={OUT_MONTHLY_CSV}")
    print(f"totals_csv={OUT_TOTALS_CSV}")


if __name__ == "__main__":
    main()
