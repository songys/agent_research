#!/usr/bin/env python3
"""
- 입력: ARXIV_SRC_GLOB의 수집 CSV, ARXIV_SRC_2026_GLOB의 2026년 JSON, verification/review_*.jsonl.
- 출력: preprocessing/output/*_2023_2026q3.csv, corpus_manifest.json, verification/corpus_counts_YYYYMMDD.json.
- 원고: 3장 자료와 표 1.
- 기간: arXiv 식별자 최초 제출 월 기준 2023-01~2026-09.
- 선정: 제목·초록의 agent 계열과 LLM 계열 용어를 함께 사용한다.
- 언어: 수집 신호에 사람 판정을 적용한다. 여러 언어를 다루면 언어별로 중복 집계한다.
- 중복: 식별자 중복과 20자 이상 정규화 제목이 같은 재제출을 통합한다.
- 국가 열은 보존하지만 현재 원고의 비교 축으로 사용하지 않는다.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_GLOB = os.environ.get(
    "ARXIV_SRC_GLOB",
    str(ROOT / "preprocessing/raw/arxiv_cs_*_20221101_20260228_enriched.csv"),
)
OUT_DIR = ROOT / "preprocessing/output"
VER_DIR = ROOT / "verification"
MAN_DIR = ROOT / "preprocessing/output"

YEAR_MIN, YEAR_MAX = 2023, 2026
DATE_MAX = "2026-09-30"      # D30: 분석 기간 2023-01-01 ~ 2026-09-30 (HCLT 2026 논문집과 같은 끝점)
# 연도 기준(D31 후보): "published" = 수집본의 published(수집 시점의 최신 판 날짜라 증분 수집마다 옛 논문이 뒤 연도로 밀린다),
# "id" = arXiv id 접두 YYMM(최초 제출 월, 수집 시점과 무관). 환경변수 ARXIV_YEAR_BASIS로 고른다.
YEAR_BASIS = os.environ.get("ARXIV_YEAR_BASIS", "id")   # D31: 기본은 id 접두(최초 제출 월)
SUFFIX = "2023_2026q3"       # 산출 파일 이름 꼬리. 이전 2023_2025 파일은 PACLIC 그림 3 재현 검사(04)용으로 남긴다
# 2026-03~09는 scripts/arxiv_collect/(D24)로 받은 OAI 수집본. 소속·국가 열이 없다(country=NaN)
SRC_2026_GLOB = os.environ.get("ARXIV_SRC_2026_GLOB", str(ROOT / "preprocessing/raw/arxiv_2026/analysis/arxiv_data/arxiv_cs_*_20260301_20260930.json"))

# --- 정의 D1: LLM 에이전트 ----------------------------------------------------
AGENT_RE = re.compile(r"\bagent(s|ic)?\b|\bmulti-agent\b")
LLM_RE = re.compile(
    r"\b(llms?|large language models?|language models?|generative ai|foundation models?|"
    r"gpt(-?[0-9o.]+)?|chatgpt|claude|gemini|llama|qwen|vlms?|vision[- ]language models?)\b"
)
LLM_AGENT_ADJ_RE = re.compile(
    r"\b(llm|language[- ]model|generative|gpt|foundation[- ]model|vlm)[- ]?(based |powered |driven |enabled )?agents?\b"
)

# --- 탐색 D17: agent 태그 논문 안의 개념 연결(concept links) — 제목+초록 정규식, 다중 라벨 ---
CONCEPTS = {
    "llm": r"\b(llms?|large language models?|language models?|gpt|chatgpt|foundation models?|generative ai|vlms?)\b",
    "physical": r"\b(embodied|robots?|robotic\w*|physical|manipulat\w*|locomotion|autonomous driving|vehicles?|drones?|sim-to-real|real-world environments?)\b",
    "multiagent_system": r"\b(multi-?agent|agent systems?|agentic systems?|collaborat\w*|coordinat\w*|cooperat\w*)\b",
    "rl": r"\b(reinforcement learning|rl|rewards?|policy optimization|markov decision|q-learning)\b",
    "simulation": r"\b(simulat\w*|social|society|societal|economic|opinion)\b",
    "tool": r"\b(tool[- ]?use|tools?|apis?|function calling|mcp)\b",
    "human": r"\b(humans?|users?|interaction|interactive|human-in-the-loop)\b",
    "safety": r"\b(safety|safe|attacks?|adversarial|security|jailbreak\w*|alignment)\b",
    "planning": r"\b(planning|planner|plans?|task decomposition)\b",
    "memory": r"\b(memory|long-term|long-context)\b",
    "evaluation": r"\b(benchmarks?|evaluat\w*)\b",
    "software": r"\b(software|code generation|coding|programming|github)\b",
    "gui_web": r"\b(gui|web agents?|web navigation|browser|mobile|app)\b",
}
CONCEPT_RE = {k: re.compile(v) for k, v in CONCEPTS.items()}

# --- 정의 D10: 대상 언어 태그 ---------------------------------------------------
LANGS = {"korean": "ko", "chinese": "zh", "japanese": "ja"}

# --- 보조: 국가 정규화 ----------------------------------------------------------
COUNTRY_MAP = {
    "south korea": "KR", "republic of korea": "KR", "korea": "KR", "korea, republic of": "KR",
    "china": "CN", "people's republic of china": "CN", "prc": "CN", "mainland china": "CN",
    "japan": "JP",
    "hong kong": "HK", "hong kong sar": "HK", "taiwan": "TW", "macau": "MO", "macao": "MO",
    "united states": "US", "usa": "US", "us": "US", "united states of america": "US", "u.s.": "US", "u.s.a.": "US",
    "united kingdom": "GB", "uk": "GB", "england": "GB",
    "the netherlands": "NL", "netherlands": "NL",
}
CJK_COUNTRIES = {"KR", "CN", "JP"}


def norm_country(x) -> str | None:
    if not isinstance(x, str) or not x.strip():
        return None
    k = x.strip().lower()
    return COUNTRY_MAP.get(k, x.strip())


def is_llm_agent(text: str) -> bool:
    t = text.lower()
    if LLM_AGENT_ADJ_RE.search(t):
        return True
    return bool(AGENT_RE.search(t)) and bool(LLM_RE.search(t))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


LABEL_TARGET = "그 언어의 텍스트를 실제 연구 대상으로 다룸"
LABEL_MULTI = "다국어 실험의 한 언어로 포함"


def load_lang_labels() -> tuple[dict, dict]:
    """사람 검수 정본(jsonl)에서 id → 최종 라벨. 재점검 파일이 본 검수 파일을 덮어쓴다(D36). 라벨이 없으면 빈 dict."""
    def latest(pattern):
        fs = sorted(glob.glob(str(VER_DIR / pattern)))
        return Path(fs[-1]) if fs else None
    p_main, p_re = latest("review_review_lang_tag_2023_2026_*.jsonl"), latest("review_review_lang_tag_recheck_*.jsonl")
    labels, meta = {}, {"main_file": None, "recheck_file": None, "n_main": 0, "n_recheck": 0, "n_recheck_changed": 0}
    if p_main is None:
        return labels, meta
    for line in p_main.read_text().splitlines():
        r = json.loads(line); labels[r["id"]] = r["final"]
    meta.update(main_file=str(p_main.relative_to(ROOT)), n_main=len(labels))
    if p_re is not None:
        n_changed = 0
        for line in p_re.read_text().splitlines():
            r = json.loads(line)
            if labels.get(r["id"]) != r["final"]:
                n_changed += 1
            labels[r["id"]] = r["final"]
        meta.update(recheck_file=str(p_re.relative_to(ROOT)), n_recheck=sum(1 for _ in p_re.read_text().splitlines()), n_recheck_changed=n_changed)
    p_fix = latest("review_review_lang_tag_corrections_*.jsonl")      # 사용자 결정 정정(D40): 재점검 뒤에 덮어쓴다
    if p_fix is not None:
        n_fix = 0
        for line in p_fix.read_text().splitlines():
            r = json.loads(line); labels[r["id"]] = r["final"]; n_fix += 1
        meta.update(corrections_file=str(p_fix.relative_to(ROOT)), n_corrections=n_fix)
    return labels, meta


def by(df, *keys):
    return {str(k) if not isinstance(k, tuple) else "|".join(map(str, k)): int(v)
            for k, v in df.groupby(list(keys)).size().items()}


def main() -> None:
    files = sorted(glob.glob(SRC_GLOB))
    if not files:
        sys.exit(f"입력 파일 없음: {SRC_GLOB}")
    usecols = ["id", "url", "title", "abstract", "published", "categories", "primary_category",
               "detected_keywords", "country", "affiliations"]
    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=usecols, encoding="utf-8-sig", low_memory=False, dtype={"id": str})
        df["id"] = df["id"].str.strip()           # id는 문자열로 고정(D6)
        # 수집본의 id 열은 일부가 숫자형 파싱으로 끝 0이 떨어져 있다(예: 2203.0486 ← 2203.04860, 7,288건).
        # url 열의 abs/ 뒤 id가 온전하므로 거기서 복원한다(D13). 버전 접미(v1 등)는 뗀다.
        from_url = df["url"].astype(str).str.extract(r"abs/([a-z\-]+/\d{7}|\d{4}\.\d{4,5})")[0]
        df["id"] = from_url.where(from_url.notna(), df["id"]).str.replace(r"v\d+$", "", regex=True)
        frames.append(df)
    files_2026 = sorted(glob.glob(SRC_2026_GLOB))
    for f in files_2026:                             # D30: 2026-03~09 OAI 수집본(JSON)을 같은 열로 맞춰 붙인다
        j = json.loads(Path(f).read_text())
        d = pd.DataFrame(j["papers"])
        d = pd.DataFrame({"id": d["id"].astype(str), "url": d["url"], "title": d["title"], "abstract": d["abstract"], "published": d["published"],
                          "categories": d["categories"].map(lambda L: " ".join(L) if isinstance(L, list) else str(L)), "primary_category": d["primary_category"],
                          "detected_keywords": d["detected_keywords"].map(lambda L: str(list(L))), "country": pd.NA, "affiliations": pd.NA})
        frames.append(d)
    files = files + files_2026
    raw = pd.concat(frames, ignore_index=True)
    n_rows_raw = len(raw)
    raw = raw.drop_duplicates("id").copy()          # 교차 등재(cross-listing)·수집본 간 중복 제거(앞 파일 우선)
    # 제목이 같은 다른 id(재제출·중복 등록)는 한 편으로 센다(사용자 지시 2026-10-14): 제목을 소문자·영숫자만으로 정규화해 같으면 id가 빠른 것만 남긴다
    tnorm = raw.title.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    dup_title_mask = tnorm.duplicated(keep=False) & (tnorm.str.len() >= 20)
    n_title_dup_groups = int(tnorm[dup_title_mask].nunique()); n_title_dup_rows = int(dup_title_mask.sum())
    raw = raw.assign(_tnorm=tnorm).sort_values("id")
    raw = raw[~(raw._tnorm.duplicated(keep="first") & (raw._tnorm.str.len() >= 20))].drop(columns="_tnorm").copy()
    n_title_dup_removed = n_title_dup_rows - n_title_dup_groups
    # 연도·월은 수집본의 published 열(D15, PACLIC 2026과 같은 기준). arXiv id 접두 연도는 민감도 점검용으로 year_id에 보존한다.
    pub = pd.to_datetime(raw["published"], errors="coerce", utc=True, format="mixed")
    raw["year"] = pub.dt.year.astype("Int64")
    raw["month"] = pub.dt.month.astype("Int64")
    new_style = raw.id.str.match(r"^\d{4}\.\d{4,5}$")
    raw["year_id"] = pd.Series(pd.NA, index=raw.index, dtype="Int64")
    raw.loc[new_style, "year_id"] = (2000 + raw.loc[new_style, "id"].str[:2].astype(int)).astype("Int64")
    n_year_mismatch = int((raw.year.notna() & raw.year_id.notna() & (raw.year != raw.year_id)).sum())
    raw["year_published"] = raw["year"]; raw["month_published"] = raw["month"]
    raw["month_id"] = pd.Series(pd.NA, index=raw.index, dtype="Int64")
    raw.loc[new_style, "month_id"] = raw.loc[new_style, "id"].str[2:4].astype(int).astype("Int64")
    if YEAR_BASIS == "id":
        # 최초 제출 월(id 접두)로 연도·월을 정하고 기간을 자른다. 옛 형식 id(새 형식이 아닌 11건)는 제외된다
        raw["year"] = raw["year_id"]; raw["month"] = raw["month_id"]
        ym = raw.year_id * 100 + raw.month_id
        base = raw[(ym >= YEAR_MIN * 100 + 1) & (ym <= int(DATE_MAX[:4]) * 100 + int(DATE_MAX[5:7]))].copy()
    else:
        base = raw[(raw.year >= YEAR_MIN) & (raw.year <= YEAR_MAX) & (pub <= pd.Timestamp(DATE_MAX + " 23:59:59", tz="UTC"))].copy()

    text = base.title.fillna("") + " " + base.abstract.fillna("")
    base["is_llm_agent"] = text.map(is_llm_agent)
    kw = base.detected_keywords.fillna("[]")
    base["agent_kw_tag"] = kw.str.contains("'agent'")
    for name, code in LANGS.items():
        base[f"lang_{code}_signal"] = kw.str.contains(f"'{name}'")
    sig_cols = [f"lang_{c}_signal" for c in LANGS.values()]
    base["cjk_signal"] = base[sig_cols].any(axis=1)
    base["target_langs_signal"] = base[sig_cols].apply(
        lambda r: ";".join(c.replace("lang_", "").replace("_signal", "") for c in sig_cols if r[c]), axis=1)
    # D36: 사람 검수 라벨을 붙여 최종 대상 언어 집합을 만든다(라벨은 LLM 에이전트 ∧ 신호 257편에만 있다)
    lang_labels, label_meta = load_lang_labels()
    base["lang_label"] = base.id.map(lang_labels).fillna("")
    base["lang_target_human"] = base.lang_label == LABEL_TARGET
    base["lang_multilingual_human"] = base.lang_label == LABEL_MULTI
    for code in LANGS.values():
        base[f"lang_{code}"] = base[f"lang_{code}_signal"] & base.lang_target_human
    lang_cols = [f"lang_{c}" for c in LANGS.values()]
    base["cjk_lang"] = base[lang_cols].any(axis=1)
    base["target_langs"] = base[lang_cols].apply(
        lambda r: ";".join(c.replace("lang_", "") for c in lang_cols if r[c]), axis=1)
    # 수집 단계의 정규화 주제군 태그(PACLIC 2026 부록 사전: llm, rag, transformer, multimodal, diffusion, reinforcement_learning,
    # agent, fine_tuning, robotics, tabular, question_answering, translation, 언어 태그). 거친 주제 층(coarse layer)으로 보존한다.
    base["kw_families"] = kw.str.replace(r"[\[\]' ]", "", regex=True)      # "llm,agent,rag" 형태
    base["country_iso"] = base.country.map(norm_country)
    base["country_missing"] = base.country_iso.isna()

    agent_all = base[base.is_llm_agent].copy()
    cjk_signal = agent_all[agent_all.cjk_signal].copy()          # 키워드 신호 257편(검수 입력 전체, 라벨 열 포함)
    cjk_lang = cjk_signal[cjk_signal.cjk_lang].copy()            # 최종 대상 언어 집합(라벨 = 실제 연구 대상)
    unreviewed = cjk_signal[cjk_signal.lang_label == ""]
    if len(unreviewed):
        print(f"경고: 신호 {len(cjk_signal)}편 중 라벨 없는 논문 {len(unreviewed)}편 → 대상 집합에서 제외됨", file=sys.stderr)

    # 탐색(D17): agent 키워드 태그 논문 전체(LLM 여부 무관)와 개념 연결 플래그. 키워드 연결망을 위해 초록을 포함한다(CC0 메타데이터).
    agent_tag = base[base.agent_kw_tag].copy()
    tl = (agent_tag.title.fillna("") + " " + agent_tag.abstract.fillna("")).str.lower()
    for k, rx in CONCEPT_RE.items():
        agent_tag[f"c_{k}"] = tl.map(lambda x: bool(rx.search(x)))

    # 공개용 열(소속 원문·개인 정보 제외). arXiv 메타데이터는 CC0. country_iso는 수집 단계의 LLM 추론값(D7 미결).
    pub_cols = ["id", "title", "abstract", "published", "year", "month", "year_id", "month_id", "year_published", "month_published", "primary_category", "categories",
                "target_langs", "lang_ko", "lang_zh", "lang_ja", "cjk_lang",
                "target_langs_signal", "lang_ko_signal", "lang_zh_signal", "lang_ja_signal", "cjk_signal",
                "lang_label", "lang_target_human", "lang_multilingual_human", "country_iso", "agent_kw_tag", "kw_families"]
    all_out = agent_all[pub_cols].sort_values(["year", "id"]).reset_index(drop=True)
    cjk_out = cjk_signal[pub_cols].sort_values(["year", "id"]).reset_index(drop=True)   # 257편 전부(라벨 증빙), cjk_lang 열이 최종 집합

    tag_cols = ["id", "title", "abstract", "year", "month", "primary_category", "country_iso", "target_langs", "cjk_lang",
                "is_llm_agent", "kw_families"] + [f"c_{k}" for k in CONCEPTS]
    tag_out = agent_tag[tag_cols].sort_values(["year", "id"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True); VER_DIR.mkdir(exist_ok=True); MAN_DIR.mkdir(exist_ok=True)
    p_all = OUT_DIR / f"corpus_llm_agent_all_{SUFFIX}.csv"
    p_cjk = OUT_DIR / f"corpus_llm_agent_cjk_lang_{SUFFIX}.csv"
    p_tag = OUT_DIR / f"corpus_agent_tag_{SUFFIX}.csv"
    all_out.to_csv(p_all, index=False); cjk_out.to_csv(p_cjk, index=False); tag_out.to_csv(p_tag, index=False)
    for stale in ["corpus_llm_agent_cjk_2023_2025.csv"]:        # D2 국가 기준 파일은 폐기(D10)
        sp = OUT_DIR / stale
        if sp.exists():
            sp.unlink()

    def lang_year(df):
        return {code: by(df[df[f"lang_{code}"]], "year") for code in LANGS.values()}

    counts = {
        "generated": date.today().isoformat(),
        "script": "preprocessing/scripts/01_build_corpus.py",
        "inputs": [{"file": os.path.basename(f), "sha256": sha256(Path(f))} for f in files],
        "definitions": {
            "llm_agent": "title+abstract: LLM_AGENT_ADJ_RE 또는 (AGENT_RE and LLM_RE)",
            "target_language": "detected_keywords in {'korean','chinese','japanese'} (제목·초록 키워드, PACLIC 2026과 동일) = 신호(lang_*_signal); 최종 집합 = 신호 ∧ 사람 라벨 '%s' (D36). '%s'은 집합 밖, 별도 집계" % (LABEL_TARGET, LABEL_MULTI),
            "country_aux": "country(PDF 첫 페이지 → 비전 LLM, 폴백 LaTeX → 텍스트 LLM) → ISO 정규화; 보조 정보",
            "period": f"{YEAR_MIN}-01-01 ~ {DATE_MAX} (D30 기간), 연도 기준 = {YEAR_BASIS} (published: 수집본 날짜, id: arXiv id 접두 최초 제출 월). 2026-03~09는 소속 국가 없음(D24)",
            "year_basis": YEAR_BASIS,
        },
        "pipeline": {
            "rows_raw_all_files": int(n_rows_raw),
            "unique_ids_all_years": int(len(raw)),
            "same_title_different_id": {"groups": n_title_dup_groups, "rows": n_title_dup_rows, "removed": n_title_dup_removed,
                                        "rule": "제목 정규화(소문자·영숫자) 20자 이상이 같으면 id가 빠른 한 편만 남김"},
            "new_style_ids": int(new_style.sum()),
            "year_published_vs_id_prefix_mismatch": n_year_mismatch,
            "unique_ids_in_period": int(len(base)),
            "unique_by_year": by(base, "year"),
            "unique_by_year_other_basis": by(base, "year_id" if YEAR_BASIS == "published" else "year_published"),
            "unique_by_year_month_2026": {str(k): int(v) for k, v in base[base.year == 2026].groupby("month").size().items()},
            "rows_2026_oai_files": int(sum(len(json.loads(Path(f).read_text())["papers"]) for f in files_2026)),
            "llm_agent_all": int(len(agent_all)),
            "llm_agent_cjk_signal": int(len(cjk_signal)),
            "llm_agent_cjk_lang": int(len(cjk_lang)),
            "llm_agent_cjk_lang_multi_tag": int((cjk_lang[lang_cols].sum(axis=1) > 1).sum()),
            "llm_agent_cjk_signal_unlabeled": int(len(unreviewed)),
            "country_fill_rate_by_year_all_papers": {str(k): round(float(v), 4) for k, v in
                                                     base.groupby("year").country_iso.apply(lambda s: s.notna().mean()).items()},
            "country_missing_in_cjk_lang": int(cjk_lang.country_missing.sum()),
        },
        "language_label_review": {                   # D36: 사람 검수 라벨(논문 단위)과 신호별 분포
            **label_meta,
            "label_counts": {str(k): int(v) for k, v in cjk_signal.lang_label.replace("", "(없음)").value_counts().items()},
            "label_by_signal": {str(k): {str(l): int(v) for l, v in g.lang_label.replace("", "(없음)").value_counts().items()}
                                for k, g in cjk_signal.groupby("target_langs_signal")},
            "signal_by_lang_total": {code: int(cjk_signal[f"lang_{code}_signal"].sum()) for code in LANGS.values()},
            "multilingual_by_lang_total": {code: int((cjk_signal[f"lang_{code}_signal"] & cjk_signal.lang_multilingual_human).sum()) for code in LANGS.values()},
            "multilingual_by_lang_year": {code: by(cjk_signal[cjk_signal[f"lang_{code}_signal"] & cjk_signal.lang_multilingual_human], "year") for code in LANGS.values()},
            "excluded_ids_by_label": {str(k): sorted(g.id.tolist()) for k, g in cjk_signal[~cjk_signal.lang_target_human].groupby(cjk_signal.lang_label.replace("", "(없음)"))},
        },
        "cjk_signal_by_lang_year": {code: by(cjk_signal[cjk_signal[f"lang_{code}_signal"]], "year") for code in LANGS.values()},
        "llm_agent_by_year": by(agent_all, "year"),
        "cjk_lang_by_year": by(cjk_lang, "year"),
        "cjk_lang_by_lang_year": lang_year(cjk_lang),
        "cjk_lang_by_lang_total": {code: int(cjk_lang[f"lang_{code}"].sum()) for code in LANGS.values()},
        "cjk_lang_by_country": {str(k): int(v) for k, v in cjk_lang.country_iso.fillna("(missing)").value_counts().items()},
        "cjk_lang_by_primary_category": by(cjk_lang, "primary_category"),
        "language_target_papers_all_topics": {       # 분모: 에이전트 무관, 그 언어를 대상으로 한 논문 전체
            code: {"total": int(base[f"lang_{code}_signal"].sum()),
                   "llm_agent": int((base[f"lang_{code}_signal"] & base.is_llm_agent).sum()),
                   "llm_agent_final": int(base[f"lang_{code}"].sum()),
                   "by_year_total": by(base[base[f"lang_{code}_signal"]], "year"),
                   "basis": "키워드 신호(검수 없음); llm_agent_final만 D36 라벨 반영"}
            for code in LANGS.values()},
        "loose_agent_tag_reference": {code: int((base[f"lang_{code}_signal"] & base.agent_kw_tag).sum()) for code in LANGS.values()},
        "agent_tag_exploration": {
            "n": int(len(agent_tag)), "by_year": by(agent_tag, "year"),
            "llm_agent_share": round(float(agent_tag.is_llm_agent.mean()), 4),
            "concept_rate": {k: round(float(agent_tag[f"c_{k}"].mean()), 4) for k in CONCEPTS},
            "concept_regex": CONCEPTS,
        },
        "country_cjk_reference_llm_agent": {          # D2(폐기)의 국가 기준 수치, 비교용
            c: by(agent_all[agent_all.country_iso == c], "year") for c in sorted(CJK_COUNTRIES)},
        "country_vs_language_overlap_llm_agent": {
            f"{c}|{code}": {"country_n": int((agent_all.country_iso == c).sum()),
                            "country_with_lang_tag": int(((agent_all.country_iso == c) & agent_all[f"lang_{code}"]).sum()),
                            "lang_n": int(agent_all[f"lang_{code}"].sum()),
                            "lang_with_country": int((agent_all[f"lang_{code}"] & (agent_all.country_iso == c)).sum()),
                            "lang_country_missing": int((agent_all[f"lang_{code}"] & agent_all.country_missing).sum())}
            for c, code in [("KR", "ko"), ("CN", "zh"), ("JP", "ja")]},
    }
    p_ver = VER_DIR / f"corpus_counts_{date.today():%Y%m%d}.json"
    p_ver.write_text(json.dumps(counts, ensure_ascii=False, indent=2))

    manifest = {
        "generated": date.today().isoformat(),
        "files": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "rows": int(len(d)),
             "key": "id", "key_unique": bool(d["id"].is_unique)}
            for p, d in [(p_all, all_out), (p_cjk, cjk_out), (p_tag, tag_out)]
        ],
    }
    (MAN_DIR / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(json.dumps({k: counts[k] for k in ["pipeline", "llm_agent_by_year", "cjk_lang_by_year",
                                              "cjk_lang_by_lang_year", "cjk_lang_by_country"]}, ensure_ascii=False, indent=2))
    print("wrote", p_all, p_cjk, p_ver, sep="\n  ")


if __name__ == "__main__":
    main()
