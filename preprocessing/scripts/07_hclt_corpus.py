#!/usr/bin/env python3
"""
07_hclt_corpus.py — HCLT(한글 및 한국어 정보처리 학술대회) 논문집 PDF에서 논문 단위 메타데이터를 뽑고 LLM 에이전트 논문을 판정한다

입력 파일
  preprocessing/raw/hclt_pdf/{2023,2024,2025,2026}hclt.pdf — 사용자가 제공한 논문집 전체 PDF(저장소에 넣지 않는다, 각 60~375 MB)
  preprocessing/output/hclt_ocr/<year>_p<page>.txt        — (있으면) 글꼴이 깨진 첫 면의 OCR 텍스트(08_hclt_ocr.py가 만든다)
출력 파일
  preprocessing/output/hclt_papers.csv            — 논문별: year, printed_page, pdf_page, title_ko(목차 기준), authors_ko(목차), title_en, abstract_ko, keywords_ko,
                                              source(text|ocr|toc_only), is_llm_agent, agent_tag, kw_families
  verification/hclt_counts_YYYYMMDD.json   — 연도별 논문 수(목차 기준), 추출 방식별 수, 에이전트 판정 수, 키워드군 비율(정본)
  analysis/output/review/review_hclt_agent.jsonl   — 사람 검수 입력(agent 단어가 있는 논문 전부)
논문 위치
  (원고 반영은 사용자 지시 후) 4.6절 후보: 국내 학회 논문과의 대조

방법
  - 논문 명부(roster)는 논문집 앞부분의 프로그램 목차(「NN. 제목 ······ 쪽」 + 저자 줄)에서 만든다. 제목·저자·인쇄 쪽수가 정본이다.
  - 인쇄 쪽수 → PDF 쪽 인덱스 오프셋은 목차 제목과 본문 첫 면 제목이 맞는 쌍의 최빈값으로 정한다.
  - 각 논문의 첫 면에서 영문 제목, 요약(「요약」과 「주제어」 사이), 주제어를 뽑는다. 2023년 논문집은 일부 첫 면의 글꼴이 깨져 텍스트가 나오지 않으므로
    OCR 텍스트(preprocessing/output/hclt_ocr/)가 있으면 그것을 쓴다. 둘 다 없으면 목차 제목만으로(source=toc_only) 판정한다.
  - PyMuPDF 추출은 한국어 어절 사이 공백을 잃는다. 정규식은 공백을 뺀 문자열에 적용한다.
  - LLM 에이전트 판정(D1의 국문 대응): (에이전트|agent) AND (LLM|언어모델|대규모언어모델|초거대언어모델|거대언어모델|생성형|GPT|large language model).
    목차 제목 + 영문 제목 + 요약 + 주제어에 적용.
  - 키워드군: PACLIC 사전 11군 + 확장 6군의 영문 정규식에 국문 대응 패턴을 더했다(FAMILIES).
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "preprocessing/raw/hclt_pdf"
OCR_DIR = ROOT / "preprocessing/output/hclt_ocr"
OUT_CSV = ROOT / "preprocessing/output/hclt_papers.csv"
REV = ROOT / "analysis/output/review"; REV.mkdir(parents=True, exist_ok=True)

AGENT_RE = re.compile(r"에이전트|agent", re.I)
LLM_RE = re.compile(r"llm|언어모델|대규모언어모델|초거대언어모델|거대언어모델|대형언어모델|생성형|gpt|largelanguagemodel|languagemodel|sllm|vlm", re.I)
FAMILIES = {
    "llm": (r"\b(llms?|large language models?|language models?|gpt|chatgpt|foundation models?|generative ai|vlms?)\b", r"언어모델|생성형|llm|gpt"),
    "rag": (r"\b(rag|retrieval[- ]augmented|retrieval)\b", r"검색증강|검색기반|리트리벌|rag"),
    "transformer": (r"\btransformers?\b", r"트랜스포머"),
    "multimodal": (r"\b(multi-?modal|vision[- ]language|image|visual)\b", r"멀티모달|다중모달|시각|이미지|비전"),
    "diffusion": (r"\bdiffusion\b", r"확산모델|디퓨전"),
    "reinforcement_learning": (r"\b(reinforcement learning|rl|reward|policy)\b", r"강화학습|보상|rlhf|grpo|dpo"),
    "robotics": (r"\b(robot\w*|embodied|manipulation)\b", r"로봇|체화|로보틱스"),
    "fine_tuning": (r"\b(fine-?tun\w+|lora|instruction tuning)\b", r"미세조정|파인튜닝|지시학습|lora"),
    "tabular": (r"\b(tabular|table|sql)\b", r"표데이터|테이블|sql|표형"),
    "question_answering": (r"\b(question answering|qa)\b", r"질의응답|질문답변|qa"),
    "translation": (r"\b(translation|machine translation)\b", r"번역|기계번역"),
    "tool": (r"\b(tool[- ]?use|tools?|apis?|function calling|mcp)\b", r"도구사용|도구호출|함수호출|도구|api|mcp"),
    "simulation": (r"\b(simulat\w*|social|society|opinion)\b", r"시뮬레이션|사회|여론|모사"),
    "human": (r"\b(humans?|users?|interaction|interactive)\b", r"사용자|인간|상호작용|사람"),
    "safety": (r"\b(safety|safe|attacks?|adversarial|security|jailbreak\w*|alignment|harmful)\b", r"안전|공격|적대적|보안|탈옥|정렬|유해"),
    "evaluation": (r"\b(benchmarks?|evaluat\w*)\b", r"벤치마크|평가"),
    "physical": (r"\b(embodied|robots?|physical|manipulat\w*|navigation|autonomous driving|vehicles?)\b", r"체화|로봇|물리|내비게이션|자율주행"),
}
TOC_RE = re.compile(r"(?<!\d)(\d{2})\.\s+(\S.{5,}?)·{3,}\s*(\d{1,4})")


def norm(s: str) -> str:
    return re.sub(r"\[.*?\]|[\s\x00-\x1f]|[^\w가-힣]", "", str(s)).lower()


def roster_from_toc(d: fitz.Document) -> list[dict]:
    """프로그램 목차에서 (제목, 인쇄 쪽수, 저자 줄)을 뽑는다. 저자 줄은 제목 줄 다음 줄이다."""
    items = []
    for i in range(0, 45):
        lines = d[i].get_text().split("\n")
        if len(TOC_RE.findall(" ".join(lines))) < 2:
            continue
        for li, line in enumerate(lines):
            m = TOC_RE.search(line)
            if not m and li > 0 and re.match(r"^\s*\d{2}\.\s", lines[li - 1]) and "·" in line:
                m = TOC_RE.search(lines[li - 1] + " " + line)      # 제목이 두 줄에 걸친 경우
            if not m:
                continue
            title = re.sub(r"\s+", " ", m.group(2)).strip()
            nxt = lines[li + 1].strip() if li + 1 < len(lines) else ""
            authors = nxt if nxt and "·" not in nxt and not re.match(r"^\s*\d{2}\.\s", nxt) else ""
            items.append({"toc_title": title, "printed_page": int(m.group(3)), "authors_ko": authors})
    seen, out = set(), []
    for it in items:
        k = (norm(it["toc_title"])[:20], it["printed_page"])
        if k not in seen:
            seen.add(k); out.append(it)
    return out


def parse_first_page(text: str) -> dict:
    """첫 면 텍스트에서 영문 제목·요약·주제어를 뽑는다. 못 찾으면 빈 문자열."""
    text = text.replace("\u200b", "").replace("\ufeff", "")
    m = re.search(r"(?im)^\s*(?:요\s*약|abstract)\s*\n", text)
    k = re.search(r"(?im)^\s*(?:주\s*제\s*어|키\s*워\s*드|key\s*words?|subject\s*words?)\s*[:：]\s*(.+)", text)
    intro = re.search(r"(?m)^\s*(?:1|I)\s*\.\s*(?:서\s*론|연\s*구\s*동\s*기)", text[m.end():]) if m else None
    intro_start = m.end() + intro.start() if intro else None
    out = {"title_en": "", "abstract_ko": "", "keywords_ko": ""}
    ends = [p for p in [k.start() if k and m and k.start() > m.end() else None, intro_start] if p is not None]
    if m and ends:
        end = min(ends)
        out["abstract_ko"] = text[m.end():end].replace("\n", " ").strip()
        if k and k.start() == end:
            out["keywords_ko"] = k.group(1).split("\n")[0].strip()
        head = [h.strip() for h in text[:m.start()].split("\n") if h.strip()]
    else:
        head = [h.strip() for h in text[:1500].split("\n") if h.strip()]
    en_lines = []
    for h in head[1:]:
        letters = sum(ch.isalpha() and ch.isascii() for ch in h)
        if "@" in h or re.search(r"university|대학교|연구원|institute|department|college|corp|lab\b", h, re.I):
            if en_lines:
                break
            continue
        if letters >= 0.6 * max(len(h), 1) and len(h) > 8 and not re.search(r"[◦∗†]", h):
            en_lines.append(h)
        elif en_lines:
            break
    out["title_en"] = " ".join(en_lines)
    return out


def normalize_ocr(txt: str) -> str:
    """OCR의 요약·주제어 표기만 정규화한다.

    - 경계가 없으면 parse_first_page가 요약을 빈 문자열로 남긴다.
    - 페이지 앞부분을 요약으로 대신 넣지 않는다.
    """
    txt = re.sub(r"(?m)^\W{0,3}[A-Za-z0-9]{0,3}\s*약\s*$", "요약", txt)
    txt = re.sub(r"주\s*제\s*어", "주제어", txt)
    txt = re.sub(r"키\s*워\s*드", "키워드", txt)
    return txt


def main() -> None:
    rows = []
    for pdf in sorted(PDF_DIR.glob("*hclt.pdf")):
        year = int(re.match(r"(\d{4})", pdf.name).group(1))
        d = fitz.open(pdf)
        roster = roster_from_toc(d)
        # 오프셋: 「요약」과 「주제어」가 함께 있는 첫 면 후보 쪽에서만 목차 제목(앞 8자)을 찾아 (PDF 쪽 − 인쇄 쪽)의 최빈값을 쓴다
        cand = [i for i in range(20, len(d)) if re.search(r"요\s*약\s*\n", d[i].get_text()) and re.search(r"\n\s*(주제어|키워드|Keywords?)\s*[:：]", d[i].get_text())]
        offs = []
        for it in roster[:60]:
            key = norm(it["toc_title"])[:8]
            if len(key) < 6:
                continue
            for i in cand:
                if key in norm(d[i].get_text()[:800]):
                    offs.append(i + 1 - it["printed_page"]); break
        off = int(pd.Series(offs).mode().iloc[0]) if offs else 20
        n_text = n_ocr = n_toc = 0
        # 1차 배치: 목차 쪽 번호 + 오프셋. 두 논문이 같은 쪽을 가리키면(목차 오타) 제목 앞 12자가 그 쪽에 있는 논문이 남고,
        # 나머지는 후보 쪽(요약+주제어가 있는 쪽) 가운데 제목이 있는 아직 안 쓴 쪽으로 옮긴다. 못 찾으면 그대로 두고 id에 꼬리표를 붙인다.
        idxs = [it["printed_page"] + off - 1 for it in roster]
        from collections import Counter
        dup_pages = {i for i, c in Counter(idxs).items() if c > 1}
        n_reloc = 0
        if dup_pages:
            used = set(idxs)
            for pos, (it, idx) in enumerate(zip(roster, idxs)):
                if idx not in dup_pages:
                    continue
                key = norm(it["toc_title"])[:12]
                if 0 <= idx < len(d) and key and key in norm(d[idx].get_text()[:800]):
                    continue  # 제목이 그 쪽에 있으면 이 논문이 맞다
                hit = [i for i in cand if i not in used and key and key in norm(d[i].get_text()[:800])]
                if hit:
                    idxs[pos] = hit[0]; used.add(hit[0]); n_reloc += 1
        seen = Counter()
        for it, idx in zip(roster, idxs):
            text = d[idx].get_text() if 0 <= idx < len(d) else ""
            seen[idx] += 1
            suffix = "" if seen[idx] == 1 else f"-{seen[idx]}"  # 끝내 못 가른 충돌(있으면 보고)
            hangul = len(re.findall(r"[가-힣]", text)); total = max(len(re.sub(r"\s", "", text)), 1)
            source = "text"
            parsed = parse_first_page(text) if hangul / total >= 0.15 else {"title_en": "", "abstract_ko": "", "keywords_ko": ""}
            if not parsed["abstract_ko"]:
                ocr = OCR_DIR / f"{year}_p{idx + 1}.txt"; ocr_kor = OCR_DIR / f"{year}_p{idx + 1}.kor.txt"
                if ocr.exists():
                    parsed = parse_first_page(normalize_ocr(ocr.read_text()))  # kor+eng 판: 영문 제목, 영문 용어
                    parsed["abstract_ocr_koreng"] = parsed["abstract_ko"]
                    if ocr_kor.exists():  # kor 판: 한글 인식이 좋아 국문 요약·주제어는 이 판을 쓴다
                        pk = parse_first_page(normalize_ocr(ocr_kor.read_text()))
                        if pk["abstract_ko"]:
                            parsed["abstract_ko"] = pk["abstract_ko"]; parsed["keywords_ko"] = pk["keywords_ko"] or parsed["keywords_ko"]
                    source = "ocr" if parsed["abstract_ko"] else "toc_only"
                else:
                    source = "toc_only"
            n_text += source == "text"; n_ocr += source == "ocr"; n_toc += source == "toc_only"
            rows.append({"id": f"hclt{year}-p{idx + 1}{suffix}", "year": year, "printed_page": it["printed_page"], "pdf_page": idx + 1,
                         "title_ko": it["toc_title"], "authors_ko": it["authors_ko"], "source": source, **parsed})
        print(f"{pdf.name}: 목차 {len(roster)}편, 오프셋 {off}, 추출 text {n_text} / ocr {n_ocr} / 목차만 {n_toc}, 제목으로 다시 찾은 쪽 {n_reloc}")
    df = pd.DataFrame(rows)
    assert df.id.is_unique, f"id 중복: {df[df.id.duplicated()].id.tolist()}"
    if "abstract_ocr_koreng" not in df:
        df["abstract_ocr_koreng"] = ""
    # 1차 판정(정규식)은 검수 입력 선별용이다. 최종 판정은 사람 검수(D23)와 형태소 기반 분석(09)이 맡는다.
    nospace = (df.title_ko.fillna("") + " " + df.abstract_ko.fillna("") + " " + df.keywords_ko.fillna("") + " " + df.abstract_ocr_koreng.fillna("")).str.replace(r"\s+", "", regex=True).str.lower()
    en = df.title_en.fillna("").str.lower()
    df["agent_tag"] = nospace.str.contains(AGENT_RE) | en.str.contains(AGENT_RE)
    df["llm_tag"] = nospace.str.contains(LLM_RE) | en.str.contains(LLM_RE)
    df["is_llm_agent"] = df.agent_tag & df.llm_tag
    # 사람 판정(D23): 접수된 검수 정본이 있으면 합친다. 없으면 열은 비워 둔다(07을 다시 돌려도 판정이 사라지지 않게)
    rev = sorted((ROOT / "verification").glob("review_review_hclt_agent_*.jsonl"))
    if rev:
        r = pd.read_json(rev[-1], lines=True)
        df["is_llm_agent_human"] = df.id.isin(set(r[r.final == "LLM 에이전트"].id))
    else:
        df["is_llm_agent_human"] = pd.NA
    df["kw_families"] = [",".join(f for f, (rx_en, rx_ko) in FAMILIES.items() if re.search(rx_en, e, re.I) or re.search(rx_ko, ns, re.I)) for ns, e in zip(nospace, en)]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    by_year = df.groupby("year").agg(papers=("id", "size"), src_text=("source", lambda s: int((s == "text").sum())), src_ocr=("source", lambda s: int((s == "ocr").sum())),
                                     src_toc_only=("source", lambda s: int((s == "toc_only").sum())), agent_word=("agent_tag", "sum"), llm_agent=("is_llm_agent", "sum"))
    counts = {"generated": date.today().isoformat(), "script": "preprocessing/scripts/07_hclt_corpus.py", "inputs": [p.name for p in sorted(PDF_DIR.glob("*hclt.pdf"))],
              "by_year": {str(y): {k: int(v) for k, v in r.items()} for y, r in by_year.iterrows()},
              "definitions": {"roster": "프로그램 목차의 「NN. 제목 ···· 쪽」 항목", "llm_agent": "(에이전트|agent) AND LLM 계열(국문·영문), 제목+영문 제목+요약+주제어",
                              "families": {f: list(v) for f, v in FAMILIES.items()}},
              "family_rate_by_year": {str(y): {f: round(float(g.kw_families.str.contains(f).mean()), 4) for f in FAMILIES} for y, g in df.groupby("year")},
              "family_rate_among_llm_agent": ({f: round(float(df[df.is_llm_agent].kw_families.str.contains(f).mean()), 4) for f in FAMILIES} if df.is_llm_agent.any() else {}),
              "llm_agent_titles": df[df.is_llm_agent][["id", "year", "title_ko", "title_en", "source"]].to_dict(orient="records")}
    (ROOT / "verification").mkdir(exist_ok=True)
    if df.is_llm_agent_human.notna().all():  # 사람 판정(D23)이 접수된 뒤에는 연도별 최종 건수를 함께 남긴다
        hb = df.groupby("year").agg(papers=("id", "size"), llm_agent_regex=("is_llm_agent", "sum"), llm_agent_human=("is_llm_agent_human", "sum"))
        counts["human_review"] = {"source": rev[-1].name, "n_reviewed": int(len(r)), "final_llm_agent": int(df.is_llm_agent_human.sum()),
                                  "regex_false_negative_upgraded": int((df.is_llm_agent_human & ~df.is_llm_agent).sum()),
                                  "by_year": {str(y): {"papers": int(v.papers), "llm_agent_regex": int(v.llm_agent_regex), "llm_agent_human": int(v.llm_agent_human),
                                                       "share_pct": round(100 * float(v.llm_agent_human) / float(v.papers), 1)} for y, v in hb.iterrows()}}
    (ROOT / "verification" / f"hclt_counts_{date.today():%Y%m%d}.json").write_text(json.dumps(counts, ensure_ascii=False, indent=1))
    print(by_year.to_string())
    rev = df[df.agent_tag].copy()
    rev["proposal"] = rev.is_llm_agent.map({True: "LLM 에이전트(정규식 선택)", False: "agent 단어만(정규식 제외)"})
    rev[["id", "year", "source", "title_ko", "title_en", "abstract_ko", "keywords_ko", "proposal"]].to_json(REV / "review_hclt_agent.jsonl", orient="records", lines=True, force_ascii=False)
    print("검수 입력", len(rev), "건")


if __name__ == "__main__":
    main()
