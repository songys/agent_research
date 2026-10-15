#!/usr/bin/env python3
"""
- 입력: preprocessing/output/hclt_papers.csv, corpus_agent_tag_2023_2026q3.csv, corpus_llm_agent_all_2023_2026q3.csv.
- 출력: analysis/output/hclt_tokens.csv, arxiv_lemma_families_2023_2026q3.csv, verification/hclt_morph_YYYYMMDD.json.
- 원고: 4.2.2 표 3. 집계 노트북: analysis/notebooks/05_hclt.ipynb.
- 한국어: Kiwi로 띄어쓰기를 복원하고 내용 형태소를 추출한다.
- 영어: spaCy en_core_web_sm의 표제어를 사용한다.
- 사전의 영어·한국어 표현에도 같은 분석을 적용하고 1~4그램으로 대조한다.
- 키워드군: 한 문서에 용어가 나타났는지를 집계한다. 여러 군에 동시에 포함될 수 있다.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import spacy
from kiwipiepy import Kiwi

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "preprocessing/output"          # 입력(정제 자료)
OUT = ROOT / "analysis/output"; OUT.mkdir(parents=True, exist_ok=True)   # 출력(분석 산출)
SUFFIX = "2023_2026q3"                         # D30 기간 말뭉치
VER = ROOT / "verification"
TODAY = date.today().strftime("%Y%m%d")

# 키워드군 사전: 07의 17군과 같은 군 이름. 값은 자연어 용어이며 아래에서 분석기로 형태 열로 바꾼다.
FAMILY_TERMS: dict[str, dict[str, list[str]]] = {
    "agent": {"en": ["agent", "agents", "agentic", "multi-agent"], "ko": ["에이전트"]},
    "llm": {"en": ["llm", "llms", "large language model", "language model", "gpt", "chatgpt", "foundation model", "generative ai", "vlm", "sllm"],
            "ko": ["언어모델", "언어 모델", "생성형", "거대 언어", "초거대", "대규모 언어", "LLM", "GPT", "sLLM", "VLM"]},
    "rag": {"en": ["rag", "retrieval-augmented", "retrieval"], "ko": ["검색증강", "검색 기반", "리트리벌", "RAG"]},
    "transformer": {"en": ["transformer", "transformers"], "ko": ["트랜스포머"]},
    "multimodal": {"en": ["multimodal", "multi-modal", "vision-language", "image", "visual"], "ko": ["멀티모달", "다중모달", "시각", "이미지", "비전"]},
    "diffusion": {"en": ["diffusion"], "ko": ["확산모델", "디퓨전"]},
    "reinforcement_learning": {"en": ["reinforcement learning", "rl", "reward", "policy"], "ko": ["강화학습", "보상", "RLHF", "GRPO", "DPO"]},
    "robotics": {"en": ["robot", "robots", "robotic", "embodied", "manipulation"], "ko": ["로봇", "체화", "로보틱스"]},
    "fine_tuning": {"en": ["fine-tuning", "fine-tune", "finetuning", "finetune", "lora", "instruction tuning"], "ko": ["미세조정", "파인튜닝", "지시학습", "LoRA"]},
    "tabular": {"en": ["tabular", "table", "tables", "sql"], "ko": ["표 데이터", "테이블", "SQL", "표형"]},
    "question_answering": {"en": ["question answering", "qa"], "ko": ["질의응답", "질문답변", "QA"]},
    "translation": {"en": ["translation", "machine translation"], "ko": ["번역", "기계번역"]},
    "tool": {"en": ["tool use", "tool", "tools", "api", "apis", "function calling", "mcp"], "ko": ["도구사용", "도구호출", "함수호출", "도구", "API", "MCP"]},
    "simulation": {"en": ["simulation", "simulate", "social", "society", "opinion"], "ko": ["시뮬레이션", "사회", "여론", "모사"]},
    "human": {"en": ["human", "humans", "user", "users", "interaction", "interactive"], "ko": ["사용자", "인간", "상호작용", "사람"]},
    "safety": {"en": ["safety", "safe", "attack", "attacks", "adversarial", "security", "jailbreak", "jailbreaking", "alignment", "harmful"],
               "ko": ["안전", "공격", "적대적", "보안", "탈옥", "정렬", "유해"]},
    "evaluation": {"en": ["benchmark", "benchmarks", "evaluation", "evaluate"], "ko": ["벤치마크", "평가"]},
    "physical": {"en": ["embodied", "robot", "robots", "physical", "manipulation", "navigation", "autonomous driving", "vehicle", "vehicles"],
                 "ko": ["체화", "로봇", "물리", "내비게이션", "자율주행"]},
}
KO_CONTENT_TAGS = ("NNG", "NNP", "NNB", "SL", "XPN", "XSN", "XR", "VV", "VA", "MAG", "SN")

_columns = pd.read_csv(DATA / "hclt_papers.csv", nrows=0).columns
if "abstract_ko" not in _columns:
    raise SystemExit("학술대회 요약은 공개 자료에서 제외합니다. 저장된 판정의 재집계는 paper_code/run.py notebooks를 사용하세요. 원문부터 재추출하려면 별도로 검수한 요약 입력이 필요합니다.")

kiwi = Kiwi()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def ko_forms(text: str) -> list[str]:
    """Kiwi 형태소 분석 → 내용 형태소의 형태(form) 열(소문자). 띄어쓰기는 영향이 없다."""
    if not isinstance(text, str) or not text.strip():
        return []
    return [t.form.lower() for t in kiwi.tokenize(text) if t.tag in KO_CONTENT_TAGS]


def ko_nouns(text: str) -> list[str]:
    """명사 열(run) 추출: 명사·외래어·접두사 형태소(NNG/NNP/SL/XPN)가 이어지는 구간을 한 열로 두고, 다른 형태소가 끼면 "|"로 끊는다.
    원문에서 맞닿은 형태소(예: 대+규모)는 한 단어로 합친다. 노트북은 이 열에서 1-gram과 끊기지 않은 2-gram의 문서 빈도를 센다
    (Kiwi가 '프레임워크'를 프레임+워크로 나누고 띄어쓰기 복원이 둘을 띄우므로, 2-gram으로 복합어를 되살린다)."""
    if not isinstance(text, str) or not text.strip():
        return []
    out, cur, prev_end = [], "", None
    for t in kiwi.tokenize(text):
        if t.tag in ("NNG", "NNP", "SL", "XPN", "XSN") and cur and prev_end is not None and t.start == prev_end:
            cur += t.form
        elif t.tag in ("NNG", "NNP", "SL", "XPN"):
            if cur: out.append(cur)
            cur = t.form
        else:
            if cur: out.append(cur)
            cur = ""
            if out and out[-1] != "|": out.append("|")
        prev_end = t.end
    if cur: out.append(cur)
    return [w for w in out if w == "|" or (len(w) > 1 and not w.isascii())]


def en_lemmas(doc) -> list[str]:
    return [t.lemma_.lower() for t in doc if not (t.is_punct or t.is_space)]


def respace(text: str, source: str) -> str:
    """띄어쓰기 복원. OCR 판은 음절 사이 공백을 먼저 없앤다(텍스트 정규화)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    if source == "ocr":
        text = re.sub(r"(?<=[가-힣.,()])\s+(?=[가-힣.,()])", "", text)
    return kiwi.space(text)


def ngrams(seq: list[str], nmax: int = 4) -> set[tuple[str, ...]]:
    return {tuple(seq[i:i + n]) for n in range(1, nmax + 1) for i in range(len(seq) - n + 1)}


def build_patterns() -> dict[str, dict[str, set[tuple[str, ...]]]]:
    """사전의 용어를 같은 분석기로 형태 열로 바꾼다(영어는 표제어 열, 한국어는 내용 형태소 열)."""
    pats = {}
    for fam, d in FAMILY_TERMS.items():
        en = {tuple(en_lemmas(nlp(t.lower()))) for t in d["en"]}
        ko = {tuple(ko_forms(t)) for t in d["ko"]}
        pats[fam] = {"en": {p for p in en if p}, "ko": {p for p in ko if p}}
    return pats


def match(pats, en_seq: list[str], ko_seq: list[str]) -> dict[str, bool]:
    g_en, g_ko = ngrams(en_seq), ngrams(ko_seq)
    return {fam: bool(g_en & p["en"]) or bool(g_ko & p["ko"]) for fam, p in pats.items()}


def main() -> None:
    pats = build_patterns()
    # ── HCLT ───────────────────────────────────────────────────────────────
    h = pd.read_csv(DATA / "hclt_papers.csv")
    for c in ("title_ko", "title_en", "abstract_ko", "keywords_ko", "abstract_ocr_koreng"):
        if c not in h:
            h[c] = ""
    h["abstract_ko_spaced"] = [respace(a, s) for a, s in zip(h.abstract_ko, h.source)]
    h["keywords_ko_spaced"] = [respace(k, s) for k, s in zip(h.keywords_ko, h.source)]
    ko_text = h.title_ko.fillna("") + ". " + h.abstract_ko_spaced + ". " + h.keywords_ko_spaced
    ko_seqs = [ko_forms(t) for t in ko_text]
    h["nouns_ko"] = [" ".join(ko_nouns(t)) for t in ko_text]
    # 영문: 영문 제목 + 국문 본문에 섞인 영문 토큰(Kiwi SL) + OCR kor+eng 판의 영문 토큰
    en_text = h.title_en.fillna("") + ". " + [" ".join(f for f in s if f.isascii()) for s in ko_seqs] + ". " + \
        [" ".join(f for f in ko_forms(t) if f.isascii()) for t in h.abstract_ocr_koreng.fillna("")]
    en_seqs = [en_lemmas(d) for d in nlp.pipe(en_text.str.lower().tolist(), batch_size=64)]
    h["lemmas_en"] = [" ".join(s) for s in en_seqs]
    fam_h = pd.DataFrame([match(pats, e, k) for e, k in zip(en_seqs, ko_seqs)], index=h.index)
    for fam in FAMILY_TERMS:
        h[f"f_{fam}"] = fam_h[fam]
    h["is_llm_agent_morph"] = h.f_agent & h.f_llm
    keep = ["id", "year", "source", "abstract_ko_spaced", "keywords_ko_spaced", "nouns_ko", "lemmas_en", "is_llm_agent_morph", "is_llm_agent_human"] + [f"f_{f}" for f in FAMILY_TERMS]
    h[keep].to_csv(OUT / "hclt_tokens.csv", index=False)

    # ── arXiv(같은 사전, 영어 표제어) ───────────────────────────────────────
    allc = pd.read_csv(DATA / f"corpus_llm_agent_all_{SUFFIX}.csv", dtype={"id": str}, usecols=["id", "year", "country_iso"])
    tag = pd.read_csv(DATA / f"corpus_agent_tag_{SUFFIX}.csv", dtype={"id": str}, usecols=["id", "title", "abstract"])
    ax = allc.merge(tag, on="id", how="left")
    assert ax.abstract.notna().all(), "arXiv 초록 결측"
    texts = (ax.title.fillna("") + ". " + ax.abstract.fillna("")).str.lower().tolist()   # 제목의 대문자 단어(Agents)를 고유명사로 보지 않도록 소문자로
    fam_rows = []
    for doc in nlp.pipe(texts, batch_size=128):
        fam_rows.append(match(pats, en_lemmas(doc), []))
    fa = pd.DataFrame(fam_rows, index=ax.index)
    out_ax = ax[["id", "year", "country_iso"]].copy()
    for fam in FAMILY_TERMS:
        out_ax[f"f_{fam}"] = fa[fam]
    out_ax.to_csv(OUT / f"arxiv_lemma_families_{SUFFIX}.csv", index=False)

    # ── 정본 ────────────────────────────────────────────────────────────────
    hum = h.is_llm_agent_human.fillna(False).astype(bool); mor = h.is_llm_agent_morph
    tp = int((hum & mor).sum()); fp = int((~hum & mor).sum()); fn = int((hum & ~mor).sum())
    hs = h[h.year.between(2023, 2025) & hum]; h26 = h[(h.year == 2026) & hum]
    kr = out_ax[out_ax.country_iso == "KR"]
    rate = lambda d, fam: round(float(d[f"f_{fam}"].mean() * 100), 1) if len(d) else None
    out = {
        "generated": TODAY, "script": "analysis/scripts/09_hclt_morph.py", "method": "kiwipiepy content-morpheme forms (ko) + spaCy en_core_web_sm lemmas (en); dictionary terms analyzed by the same tools; n-gram (<=4) exact match; no regex for keyword matching",
        "versions": {"kiwipiepy": __import__("kiwipiepy").__version__, "spacy": spacy.__version__, "en_core_web_sm": nlp.meta["version"]},
        "family_terms": FAMILY_TERMS,
        "hclt": {
            "papers_by_year": {str(y): int(n) for y, n in h.groupby("year").size().items()},
            "llm_agent_human_by_year": {str(y): int(n) for y, n in h[hum].groupby("year").size().items()},
            "llm_agent_morph_by_year": {str(y): int(n) for y, n in h[mor].groupby("year").size().items()},
            "morph_vs_human": {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                               "fn_ids": h[hum & ~mor].id.tolist(), "fp_ids": h[~hum & mor].id.tolist()},
        },
        "n": {"arxiv_world": int(len(out_ax)), "arxiv_KR": int(len(kr)), "hclt_2023_2025": int(len(hs)), "hclt_2026": int(len(h26))},
        "family_rate_pct": {fam: {"arxiv_world": rate(out_ax, fam), "arxiv_KR": rate(kr, fam), "hclt_2023_2025": rate(hs, fam), "hclt_2026": rate(h26, fam),
                                  "n_hclt_2023_2025": int(hs[f"f_{fam}"].sum()), "n_hclt_2026": int(h26[f"f_{fam}"].sum())} for fam in FAMILY_TERMS},
    }
    p = VER / f"hclt_morph_{TODAY}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("HCLT 형태소 판정 vs 사람:", out["hclt"]["morph_vs_human"])
    print("연도별 사람/형태소:", out["hclt"]["llm_agent_human_by_year"], out["hclt"]["llm_agent_morph_by_year"])
    print(pd.DataFrame(out["family_rate_pct"]).T.to_string())
    print("saved", p)


if __name__ == "__main__":
    main()
