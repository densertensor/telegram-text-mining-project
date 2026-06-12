#!/usr/bin/env python3
"""Фильтрация постов для части «исполнители» (по лекалам filter_putin_posts_from_sqlite.py).

Из posts_raw извлекаются три кейс-корпуса с метками объектов-агрегатов:
  A   — ведомства-исполнители (Роскомнадзор/РКН/инфраструктура, Минцифры, Шадаев,
        депутаты-комментаторы Клишас/Горелкин/Свинцов/Боярский, Мизулина);
  B2  — «руководство без Путина» (Кремль, АП/Вайно/Кириенко, Песков, Совбез/Патрушев, Медведев);
  C   — «безличный» дискурс об ограничениях (платформа × ограничение, без акторов и без
        президентского запроса);
опциональные уровни: GOV (Мишустин/кабмин), PARL (Володин/Матвиенко),
AGENCIES (ФАС, Генпрокуратура).

Гарды основаны на профилировании 2026-06-10 (outputs/executor_query_proposal.md):
вырезание подписных футеров до матчинга, исключение рейтинг-листов, страновые фильтры
(CLDR через Babel + иностр. прилагательные/лидеры), омонимия (свинцовый, Боярский-актёр,
казанский кремль, Совбез ООН, Дмитрий Патрушев, Владимир Кириенко и пр.).
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

CUTOFF_DEFAULT = "2026-01-16"


# ---------------------------------------------------------------------------
# Нормализация и токенизация (идентично президентской части)
# ---------------------------------------------------------------------------

def norm_text(s: str) -> str:
    return s.lower().replace("ё", "е")


def tokenize(s: str) -> list[str]:
    tokens: list[str] = []
    for t in razdel_tokenize(s):
        normalized = norm_text(t.text)
        if any(ch.isalnum() for ch in normalized):
            tokens.append(normalized)
    return tokens


@lru_cache(maxsize=200_000)
def lemma_ru_token(token: str) -> str:
    if not token:
        return token
    parsed = MORPH.parse(token)
    if not parsed:
        return token
    return norm_text(parsed[0].normal_form)


# ---------------------------------------------------------------------------
# Страновой индекс (CLDR, как в президентской части) + иностранные маркеры
# ---------------------------------------------------------------------------

def load_country_index_ru() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    ru = Locale.parse("ru")
    territories = ru.territories
    stop_words = {
        "и", "республика", "федерация", "королевство", "штаты", "острова",
        "остров", "демократическая", "народная", "соединенные",
        "центральноафриканская",
    }
    exact_to_countries: dict[str, set[str]] = defaultdict(set)
    lemma_to_countries: dict[str, set[str]] = defaultdict(set)
    ambiguous_tokens = {"того"}
    for code, name in territories.items():
        if not (len(code) == 2 and code.isalpha()):
            continue
        if code.upper() == "RU":
            continue
        country_name = str(name)
        if "неизвестн" in norm_text(country_name):
            continue
        tokens = [
            tok for tok in tokenize(country_name)
            if len(tok) >= 4 and tok not in stop_words
        ]
        if not tokens:
            continue
        core_token = tokens[-1]
        if core_token in ambiguous_tokens:
            continue
        exact_to_countries[core_token].add(country_name)
        lemma_to_countries[lemma_ru_token(core_token)].add(country_name)

    for tok in ("россия", "российская", "российской", "рф", "русский", "русская"):
        exact_to_countries.pop(tok, None)
        lemma_to_countries.pop(lemma_ru_token(tok), None)

    def add_alias(alias: str, country_name: str) -> None:
        key = norm_text(alias)
        exact_to_countries[key].add(country_name)
        lemma_to_countries[lemma_ru_token(key)].add(country_name)

    add_alias("сша", "США")
    add_alias("америка", "США")
    add_alias("британия", "Великобритания")
    add_alias("фрг", "Германия")
    add_alias("оон", "ООН")
    add_alias("снбо", "Украина")
    add_alias("ес", "Евросоюз")
    add_alias("евросоюз", "Евросоюз")
    return dict(exact_to_countries), dict(lemma_to_countries)


EXACT_COUNTRY, LEMMA_COUNTRY = load_country_index_ru()

# Иностранные прилагательные/лидеры (страновой индекс CLDR — существительные,
# прилагательные вида «украинского кабмина» он не ловит).
FOREIGN_ADJ_PREFIXES = (
    "украинск", "киевск", "американск", "французск", "немецк", "британск",
    "польск", "болгарск", "израильск", "иранск", "турецк", "казахстанск",
    "белорусск", "армянск", "азербайджанск", "молдавск", "грузинск",
    "эстонск", "латвийск", "литовск", "сирийск", "венесуэльск", "тайваньск",
)
FOREIGN_LEADER_PREFIXES = (
    "зеленск", "байден", "трамп", "макрон", "эрдоган", "алиев", "пашинян",
    "лукашенк", "асад", "мадуро", "дуда", "науседа", "санду", "вучич",
)


def country_near(tokens: list[str], idx: int, window: int) -> str | None:
    left = max(0, idx - window)
    right = min(len(tokens), idx + window + 1)
    ambiguous = {"того"}
    for w in tokens[left:right]:
        if w in ambiguous:
            continue
        cs = EXACT_COUNTRY.get(w)
        if cs and len(cs) == 1:
            return next(iter(cs))
        cs = LEMMA_COUNTRY.get(lemma_ru_token(w))
        if cs and len(cs) == 1:
            return next(iter(cs))
    return None


def foreign_marker_near(tokens: list[str], idx: int, window: int) -> str | None:
    c = country_near(tokens, idx, window)
    if c:
        return c
    left = max(0, idx - window)
    right = min(len(tokens), idx + window + 1)
    for w in tokens[left:right]:
        for p in FOREIGN_ADJ_PREFIXES:
            if w.startswith(p):
                return f"adj:{p}"
        for p in FOREIGN_LEADER_PREFIXES:
            if w.startswith(p):
                return f"leader:{p}"
    return None


# ---------------------------------------------------------------------------
# Футеры и рейтинг-листы
# ---------------------------------------------------------------------------

FOOTER_LINE_RE = re.compile(
    r"(подпис(?:аться|ывайся|ывайтесь|ка|чик)|t\.me/|max\.ru|"
    r"кана[лм]\w*\s+в\s+max|мы\s+в\s+max|соловьев\s+в\s+max|"
    r"\bбуст\w*|\bboost|вступай\w*|наш\s+канал|прислать\s+новост|"
    r"обратная\s+связь|поддержать\s+(?:нас|канал)|резервн\w+\s+канал)",
    re.IGNORECASE,
)


def strip_footer_lines(text: str) -> tuple[str, int]:
    """Удаляет навигационные/подписные строки до матчинга терминов.

    Второй проход: голая короткая строка (<=2 токенов) по соседству с подписной
    строкой — это кросс-промо названия канала («РОСКОМНАДЗОР\n\nПодписаться на
    канал»), а не упоминание актора; тоже футер. Подпись источника в конце
    обычного текста («…сбой на сети. Роскомнадзор» без подписной строки рядом)
    при этом сохраняется.
    """
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


TOP_LIST_RE = re.compile(r"топ\s*[-–—]?\s*\d{2,3}", re.IGNORECASE)


def is_rating_list(norm: str, raw: str) -> bool:
    """ТОП-N подборки каналов: имя актора там — просто строка списка."""
    if TOP_LIST_RE.search(norm) and ("канал" in norm):
        return True
    if raw.count("@") >= 8 and ("канал" in norm) and ("подборк" in norm or "топ" in norm or "рейтинг" in norm):
        return True
    return False


# ---------------------------------------------------------------------------
# Помощники окон/префиксов
# ---------------------------------------------------------------------------

def prefix_positions(tokens: list[str], prefixes: tuple[str, ...]) -> list[int]:
    out = []
    for i, t in enumerate(tokens):
        for p in prefixes:
            if t.startswith(p):
                out.append(i)
                break
    return out


def window_has_prefix(tokens: list[str], idx: int, radius: int, prefixes: tuple[str, ...]) -> bool:
    left = max(0, idx - radius)
    right = min(len(tokens), idx + radius + 1)
    for t in tokens[left:right]:
        for p in prefixes:
            if t.startswith(p):
                return True
    return False


def window_has_exact(tokens: list[str], idx: int, radius: int, words: set[str]) -> bool:
    left = max(0, idx - radius)
    right = min(len(tokens), idx + radius + 1)
    return any(t in words for t in tokens[left:right])


def pair_positions(tokens: list[str], a_prefixes: tuple[str, ...], b_prefixes: tuple[str, ...], gap: int = 2) -> list[int]:
    """Позиции i, где tokens[i] начинается с a_*, а в пределах gap справа есть b_*."""
    out = []
    for i, t in enumerate(tokens):
        if not any(t.startswith(p) for p in a_prefixes):
            continue
        for j in range(i + 1, min(len(tokens), i + 1 + gap)):
            if any(tokens[j].startswith(p) for p in b_prefixes):
                out.append(i)
                break
    return out


# ---------------------------------------------------------------------------
# Президентский запрос (для исключения из корпуса C — это другой корпус)
# ---------------------------------------------------------------------------

def matches_president_query(tokens: list[str]) -> bool:
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


# ---------------------------------------------------------------------------
# Термы-матчеры. Каждый возвращает (status, detail):
#   status: "match" | "none" | "excluded"; detail — имя гарда при исключении.
# ---------------------------------------------------------------------------

BLOCKING_CTX_PREFIXES = (
    "телеграм", "telegram", "мессенджер", "whatsapp", "вотсап", "ватсап",
    "ркн", "роскомнадзор", "блокир", "заблокир", "замедл", "интернет",
    "vpn", "впн", "госдум", "комитет", "законопроект", "лдпр", "депутат",
)


def match_rkn_full(tokens):
    for t in tokens:
        if t.startswith("роскомнадзор") or t.startswith("роскомпозор"):
            return "match", None
    return "none", None


def match_rkn_abbr(tokens):
    hit = None
    for i, t in enumerate(tokens):
        if t == "ркн":
            hit = i
            if not window_has_prefix(tokens, i, 12, ("ракет", "космодром", "носител")):
                return "match", None
    if hit is not None:
        return "excluded", "rkn_space_homonym"
    return "none", None


def match_rkn_infra(tokens):
    for t in tokens:
        if t in ("грчц", "тспу", "ссоп"):
            return "match", None
    if pair_positions(tokens, ("радиочастотн",), ("центр",), gap=2):
        return "match", None
    return "none", None


MINCIFRY_REGIONAL_LEMMAS = {"область", "республика", "край", "округ", "регион"}


def match_mincifry(tokens):
    idxs = prefix_positions(tokens, ("минцифр", "минсвяз"))
    idxs += pair_positions(tokens, ("министерств",), ("цифров",), gap=2)
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        fm = foreign_marker_near(tokens, i, 3)
        if fm:
            excl = f"mincifry_foreign:{fm}"
            continue
        left = max(0, i - 3)
        right = min(len(tokens), i + 4)
        if any(lemma_ru_token(t) in MINCIFRY_REGIONAL_LEMMAS for t in tokens[left:right]):
            excl = "mincifry_regional"
            continue
        return "match", None
    return "excluded", excl


def simple_prefix_matcher(prefix: str):
    def f(tokens):
        return ("match", None) if any(t.startswith(prefix) for t in tokens) else ("none", None)
    return f


SVINTSOV_AMBIG = {"свинцова", "свинцову", "свинцове", "свинцовым", "свинцовой"}


def match_svintsov(tokens):
    ambig_seen = False
    for i, t in enumerate(tokens):
        if t == "свинцов":
            return "match", None
        if t in SVINTSOV_AMBIG:
            ambig_seen = True
            if window_has_prefix(tokens, i, 10, BLOCKING_CTX_PREFIXES + ("андре",)):
                return "match", None
    if ambig_seen:
        return "excluded", "svintsov_adjective_ambiguous"
    return "none", None


def match_boyarsky(tokens):
    idxs = prefix_positions(tokens, ("боярск",))
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        if window_has_prefix(tokens, i, 2, ("михаил", "елизавет", "лиза", "дарь")):
            excl = "boyarsky_other_person"
            continue
        if i + 1 < len(tokens) and tokens[i + 1].startswith("дум"):
            excl = "boyarsky_historic"  # «боярская дума»
            continue
        if window_has_prefix(tokens, i, 2, ("быт", "палат", "сын", "род", "лагер")):
            excl = "boyarsky_historic"
            continue
        if window_has_prefix(tokens, i, 3, ("серге", "депутат")):
            return "match", None
        if window_has_prefix(tokens, i, 10, BLOCKING_CTX_PREFIXES + ("информполитик", "it", "айти", "max")):
            return "match", None
        excl = "boyarsky_no_context"
    return "excluded", excl


def match_mizulina(tokens):
    idxs = prefix_positions(tokens, ("мизулин",))
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        if window_has_prefix(tokens, i, 2, ("елен", "никола")):
            excl = "mizulina_other_person"
            continue
        return "match", None
    return "excluded", excl


KREMLIN_CITY_PREFIXES = (
    "казан", "нижегород", "новгород", "тульск", "псковск", "астрахан",
    "ростовск", "коломен", "измайлов", "рязан", "смолен", "тоболь",
    "зарайск", "углич", "вологод", "суздал", "волоколам", "александровск",
)


def match_kreml(tokens):
    idxs = prefix_positions(tokens, ("кремл",))
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        if window_has_prefix(tokens, i, 4, KREMLIN_CITY_PREFIXES):
            excl = "kreml_city"
            continue
        if window_has_prefix(tokens, i, 1, ("прачк",)):
            excl = "kreml_channel_name"
            continue
        return "match", None
    return "excluded", excl


def match_ap_phrase(tokens):
    idxs = pair_positions(tokens, ("администрац",), ("президент",), gap=2)
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        # якорим страновой гард на токене «президент*» внутри фразы
        j = i
        for k in range(i + 1, min(len(tokens), i + 3)):
            if tokens[k].startswith("президент"):
                j = k
                break
        fm = foreign_marker_near(tokens, j, 2)
        if fm:
            excl = f"ap_foreign:{fm}"
            continue
        return "match", None
    return "excluded", excl


def match_ap_abbr(tokens):
    hit = False
    for i, t in enumerate(tokens):
        if t == "ап":
            hit = True
            if window_has_prefix(tokens, i, 10, ("кремл", "администрац", "кириенко", "внутрипол", "куратор", "президент")):
                return "match", None
    if hit:
        return "excluded", "ap_abbr_no_context"
    return "none", None


def match_vaino(tokens):
    idxs = prefix_positions(tokens, ("вайно",))
    if not idxs:
        return "none", None
    for i in idxs:
        if window_has_prefix(tokens, i, 10, ("администрац", "кремл", "ап", "глав", "президент")):
            return "match", None
    return "excluded", "vaino_no_context"


def match_kirienko(tokens):
    idxs = prefix_positions(tokens, ("кириенко",))
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        if window_has_prefix(tokens, i, 2, ("владимир",)):
            excl = "kirienko_vk_son"
            continue
        if window_has_exact(tokens, i, 6, {"vk", "вконтакте"}):
            excl = "kirienko_vk_son"
            continue
        return "match", None
    return "excluded", excl


PESKOV_FORMS = {"песков", "пескова", "пескову", "пескове", "песковым"}


def match_peskov(tokens):
    for t in tokens:
        if t in PESKOV_FORMS:
            return "match", None
    return "none", None


def match_sovbez(tokens):
    idxs = prefix_positions(tokens, ("совбез",))
    idxs += pair_positions(tokens, ("совет",), ("безопасност",), gap=2)
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        fm = foreign_marker_near(tokens, i, 4)
        if fm:
            excl = f"sovbez_foreign:{fm}"
            continue
        if window_has_prefix(tokens, i, 4, ("белог",)) and window_has_prefix(tokens, i, 4, ("дом",)):
            excl = "sovbez_white_house"
            continue
        return "match", None
    return "excluded", excl


def match_patrushev(tokens):
    idxs = prefix_positions(tokens, ("патрушев",))
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        if window_has_prefix(tokens, i, 2, ("дмитри",)):
            excl = "patrushev_son"
            continue
        return "match", None
    return "excluded", excl


def match_medvedev(tokens):
    idxs = prefix_positions(tokens, ("медведев",))
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        if window_has_prefix(tokens, i, 2, ("даниил", "данил")):
            excl = "medvedev_tennis"
            continue
        return "match", None
    return "excluded", excl


def match_fas(tokens):
    hit = False
    for i, t in enumerate(tokens):
        if t == "фас":
            hit = True
            if window_has_prefix(tokens, i, 10, ("антимонопольн", "служб", "штраф", "возбуд", "тариф", "реклам", "оператор", "монопол", "картел", "сговор")):
                return "match", None
    if hit:
        return "excluded", "fas_no_context"
    return "none", None


def match_genprok(tokens):
    idxs = prefix_positions(tokens, ("генпрокуратур",))
    idxs += pair_positions(tokens, ("генеральн",), ("прокуратур",), gap=2)
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        fm = foreign_marker_near(tokens, i, 3)
        if fm:
            excl = f"genprok_foreign:{fm}"
            continue
        return "match", None
    return "excluded", excl


def match_kabmin(tokens):
    idxs = prefix_positions(tokens, ("кабмин",))
    if not idxs:
        return "none", None
    excl = None
    for i in idxs:
        fm = foreign_marker_near(tokens, i, 3)
        if fm:
            excl = f"kabmin_foreign:{fm}"
            continue
        return "match", None
    return "excluded", excl


# (term_name, corpus, matcher)
TERM_REGISTRY = [
    ("roskomnadzor", "A", match_rkn_full),
    ("rkn_abbr", "A", match_rkn_abbr),
    ("rkn_infra", "A", match_rkn_infra),
    ("mincifry", "A", match_mincifry),
    ("shadaev", "A", simple_prefix_matcher("шадаев")),
    ("klishas", "A", simple_prefix_matcher("клишас")),
    ("gorelkin", "A", simple_prefix_matcher("горелкин")),
    ("svintsov", "A", match_svintsov),
    ("boyarsky", "A", match_boyarsky),
    ("mizulina", "A", match_mizulina),
    ("kreml", "B2", match_kreml),
    ("adm_prezidenta", "B2", match_ap_phrase),
    ("ap_abbr", "B2", match_ap_abbr),
    ("vaino", "B2", match_vaino),
    ("kirienko", "B2", match_kirienko),
    ("peskov", "B2", match_peskov),
    ("sovbez", "B2", match_sovbez),
    ("patrushev", "B2", match_patrushev),
    ("medvedev", "B2", match_medvedev),
    ("mishustin", "GOV", simple_prefix_matcher("мишустин")),
    ("kabmin", "GOV", match_kabmin),
    ("volodin", "PARL", simple_prefix_matcher("володин")),
    ("matvienko", "PARL", simple_prefix_matcher("матвиенко")),
    ("fas", "AGENCIES", match_fas),
    ("genprok", "AGENCIES", match_genprok),
]


# ---------------------------------------------------------------------------
# Корпус C: безличный дискурс «платформа × ограничение»
# ---------------------------------------------------------------------------

PLATFORM_PREFIXES = ("телеграм", "telegram", "мессенджер", "ютуб", "youtube",
                     "whatsapp", "вотсап", "ватсап", "рунет")
PLATFORM_EXACT = {"vpn", "впн", "тг"}
RESTRICT_FIN_PREFIXES = ("счет", "счёт", "актив", "карт", "транзакц", "санкци")
RESTRICT_MIL_PREFIXES = ("позиц", "войск", "групп", "окружен", "гарнизон", "плацдарм", "котл")


def c_corpus_match(tokens) -> tuple[bool, str | None]:
    platform_idx = [i for i, t in enumerate(tokens)
                    if t in PLATFORM_EXACT or any(t.startswith(p) for p in PLATFORM_PREFIXES)]
    if not platform_idx:
        return False, None

    restrict_idx: list[int] = []
    excl = None
    for i, t in enumerate(tokens):
        kind = None
        if t.startswith("блокир") or t.startswith("заблокир") or t.startswith("разблокир"):
            kind = "block"
        elif t.startswith("замедл"):
            kind = "slow"
        elif t.startswith("ограничен") or t.startswith("ограничив"):
            kind = "restrict"
        elif t.startswith("отключ"):
            kind = "off"
        elif t.startswith("цензур"):
            kind = "cenz"
        elif t == "шатдаун":
            kind = "shutdown"
        if kind is None:
            continue
        if kind == "block" and window_has_prefix(tokens, i, 3, RESTRICT_FIN_PREFIXES):
            excl = "c_block_financial"
            continue
        if kind == "block" and window_has_prefix(tokens, i, 4, RESTRICT_MIL_PREFIXES):
            excl = "c_block_military"
            continue
        if kind in ("restrict", "off") and not any(abs(i - j) <= 12 for j in platform_idx):
            excl = f"c_{kind}_far_from_platform"
            continue
        fm = foreign_marker_near(tokens, i, 4)
        if fm:
            excl = f"c_foreign:{fm}"
            continue
        restrict_idx.append(i)

    if not restrict_idx:
        return False, excl
    # близость платформы и ограничения: длинные дайджесты с упоминанием мимоходом отсечь
    if any(abs(i - j) <= 40 for i in restrict_idx for j in platform_idx):
        return True, None
    return False, "c_far_cooccurrence"


# ---------------------------------------------------------------------------
# Пре-скрин (дёшево по подстрокам/регэкспу, до токенизации)
# ---------------------------------------------------------------------------

ACTOR_PRESCREEN_SUBSTR = (
    "роскомнадзор", "роскомпозор", "грчц", "тспу", "ссоп", "радиочастотн",
    "минцифр", "минсвяз", "шадаев", "клишас", "горелкин", "свинцов",
    "боярск", "мизулин", "генпрокуратур", "мишустин", "кабмин", "володин",
    "матвиенко", "кремл", "администрац", "вайно", "кириенко", "песков",
    "совбез", "патрушев", "медведев",
)
WORD_PRESCREEN_RE = re.compile(r"\b(ркн|ап|фас|vpn|впн|тг)\b")
PAIR_A_SUBSTR = ("министерств",)  # министерство цифрового...
PAIR_B_SUBSTR = ("цифров",)
SOVET_PAIR = ("совет", "безопасност")
RESTRICT_SUBSTR = ("блокир", "замедл", "ограничен", "ограничив", "отключ", "цензур", "шатдаун")
PLATFORM_SUBSTR = ("телеграм", "telegram", "мессенджер", "ютуб", "youtube",
                   "whatsapp", "вотсап", "ватсап", "рунет", "vpn", "впн")


def prescreen(norm: str) -> bool:
    if any(s in norm for s in ACTOR_PRESCREEN_SUBSTR):
        return True
    if WORD_PRESCREEN_RE.search(norm):
        return True
    if all(s in norm for s in SOVET_PAIR):
        return True
    if any(a in norm for a in PAIR_A_SUBSTR) and any(b in norm for b in PAIR_B_SUBSTR):
        return True
    if any(r in norm for r in RESTRICT_SUBSTR) and any(p in norm for p in PLATFORM_SUBSTR):
        return True
    return False


# ---------------------------------------------------------------------------
# Вспомогательное: views, отчёты, график
# ---------------------------------------------------------------------------

def views_to_int(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.replace(" ", "").strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def snippet(text: str, limit: int = 180) -> str:
    s = " ".join(text.split())
    return s[: limit - 3] + "..." if len(s) > limit else s


def write_exclusions_report(counter: Counter, examples: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "guard", "count", "examples"])
        for (term, guard), cnt in sorted(counter.items(), key=lambda x: -x[1]):
            w.writerow([term, guard, cnt, json.dumps(examples.get((term, guard), [])[:8], ensure_ascii=False)])


def build_plot(date_counts: dict[str, Counter], out_png: Path, out_csv: Path, cutoffs: list[date]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_dates = sorted({d for c in date_counts.values() for d in c})
    if not all_dates:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    series = sorted(date_counts.keys())
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("date," + ",".join(series) + "\n")
        for d in all_dates:
            f.write(d.isoformat() + "," + ",".join(str(date_counts[s].get(d, 0)) for s in series) + "\n")

    plt.figure(figsize=(14, 6))
    x = [d.isoformat() for d in all_dates]
    for s in series:
        plt.plot(x, [date_counts[s].get(d, 0) for d in all_dates], linewidth=1.2, label=s)
    for c in sorted(cutoffs):
        plt.axvline(c.isoformat(), linestyle="--", linewidth=1.4, color="black", label=c.isoformat())
    plt.title("Корпуса части «исполнители» по датам (с отсечкой)")
    plt.xlabel("Дата")
    plt.ylabel("Постов в день")
    plt.grid(True, alpha=0.3)
    step = max(1, len(x) // 15)
    plt.xticks(x[::step], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Фильтрация корпусов A/B2/C части «исполнители».")
    p.add_argument("--db-path", default="patriot_channels_posts_20260423_233414.sqlite")
    p.add_argument("--out-jsonl", default="outputs/executors_selection_20260423_233414.jsonl")
    p.add_argument("--reports-dir", default="outputs")
    p.add_argument("--cutoffs", default=CUTOFF_DEFAULT)
    p.add_argument("--log-every", type=int, default=200000)
    p.add_argument("--limit", type=int, default=0, help="Обработать только первые N строк (смоук-тест).")
    args = p.parse_args()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    cutoffs = [date.fromisoformat(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    conn = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    sql = "SELECT id, channel_username, payload_json FROM posts_raw"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    cur.execute(sql)

    excl_counter: Counter = Counter()
    excl_examples: dict = defaultdict(list)
    term_counter: Counter = Counter()
    corpus_counter: Counter = Counter()
    date_counts: dict[str, Counter] = {"A": Counter(), "B2": Counter(), "C": Counter()}
    rating_list_skipped = 0
    footer_lines_total = 0
    views_filtered = 0
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
            if not prescreen(norm_full):
                continue

            clean_text, removed_lines = strip_footer_lines(text)
            footer_lines_total += removed_lines
            norm = norm_text(clean_text)
            if not prescreen(norm):
                # все совпадения сидели в футере
                excl_counter[("_post", "footer_only_match")] += 1
                if len(excl_examples[("_post", "footer_only_match")]) < 8:
                    excl_examples[("_post", "footer_only_match")].append(snippet(text))
                continue
            if is_rating_list(norm, clean_text):
                rating_list_skipped += 1
                excl_counter[("_post", "rating_list")] += 1
                if len(excl_examples[("_post", "rating_list")]) < 8:
                    excl_examples[("_post", "rating_list")].append(snippet(text))
                continue

            tokens = tokenize(clean_text)
            objects: dict[str, list[str]] = defaultdict(list)
            for term, corpus, matcher in TERM_REGISTRY:
                if term == "medvedev" and (ch_user or "") == "medvedev_note":
                    excl_counter[(term, "self_channel")] += 1
                    continue
                status, guard = matcher(tokens)
                if status == "match":
                    objects[corpus].append(term)
                    term_counter[term] += 1
                elif status == "excluded" and guard:
                    excl_counter[(term, guard)] += 1
                    if len(excl_examples[(term, guard)]) < 8:
                        excl_examples[(term, guard)].append(snippet(clean_text))

            is_c = False
            if not objects and not matches_president_query(tokens):
                is_c, c_guard = c_corpus_match(tokens)
                if not is_c and c_guard:
                    excl_counter[("_c", c_guard)] += 1
                    if len(excl_examples[("_c", c_guard)]) < 8:
                        excl_examples[("_c", c_guard)].append(snippet(clean_text))

            if not objects and not is_c:
                continue

            v = views_to_int(payload.get("views"))
            if v is None or v < 1:
                views_filtered += 1
                continue

            rec = dict(payload)
            rec["_objects"] = {k: sorted(set(vv)) for k, vv in objects.items()}
            rec["_corpus_c"] = bool(is_c)
            rec["_footer_lines_removed"] = removed_lines
            # Плоские поля — для прямой загрузки в pandas/parquet на этапе 2.
            for corpus in ("A", "B2", "GOV", "PARL", "AGENCIES"):
                rec[f"_obj_{corpus}"] = bool(objects.get(corpus))
            rec["_obj_C"] = bool(is_c)
            rec["_terms"] = ",".join(sorted({t for vv in objects.values() for t in vv}))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

            ds = payload.get("date")
            d = None
            if ds:
                try:
                    d = datetime.fromisoformat(ds).date()
                except ValueError:
                    d = None
            for corpus in ("A", "B2"):
                if objects.get(corpus):
                    corpus_counter[corpus] += 1
                    if d:
                        date_counts[corpus][d] += 1
            for corpus in ("GOV", "PARL", "AGENCIES"):
                if objects.get(corpus):
                    corpus_counter[corpus] += 1
            if is_c:
                corpus_counter["C"] += 1
                if d:
                    date_counts["C"][d] += 1

    conn.close()

    print("=== RESULT ===")
    print(f"total_scanned: {total}")
    print(f"written (views>=1): {written}")
    print(f"views_filtered_out: {views_filtered}")
    print(f"rating_lists_skipped: {rating_list_skipped}")
    print(f"footer_lines_removed_total: {footer_lines_total}")
    print("corpus_counts:", dict(corpus_counter))
    print("top_terms:", term_counter.most_common(30))

    write_exclusions_report(excl_counter, excl_examples,
                            reports_dir / "executors_exclusions_report.csv")
    print(f"exclusions_report: {reports_dir / 'executors_exclusions_report.csv'}")

    summary = {
        "total_scanned": total,
        "written": written,
        "views_filtered_out": views_filtered,
        "rating_lists_skipped": rating_list_skipped,
        "footer_lines_removed_total": footer_lines_total,
        "corpus_counts": dict(corpus_counter),
        "term_counts": dict(term_counter),
        "exclusion_counts": {f"{t}|{g}": c for (t, g), c in excl_counter.items()},
        "cutoffs": [c.isoformat() for c in cutoffs],
    }
    (reports_dir / "executors_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"summary: {reports_dir / 'executors_selection_summary.json'}")

    if not args.limit:
        build_plot(date_counts,
                   reports_dir / "executors_selection_by_date_cutoffs.png",
                   reports_dir / "executors_selection_by_date.csv",
                   cutoffs)
        print("plot/csv written")


if __name__ == "__main__":
    main()
