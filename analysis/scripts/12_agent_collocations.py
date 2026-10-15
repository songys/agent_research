#!/usr/bin/env python3
"""
- 입력: preprocessing/output/corpus_llm_agent_all_2023_2026q3.csv.
- 출력: analysis/output/cache/agent_colloc_2023_2026q3.jsonl, verification/agent_colloc_extract_YYYYMMDD.json.
- 원고: 4.1.1 인접 연어. 집계 노트북: analysis/notebooks/07_agent_collocations.ipynb.
- 앞말: 명사·형용사·분사·합성어. 뒷말: 명사·합성어.
- 앞말의 한정사는 건너뛰고 multi-agent 같은 합성어의 접두를 포함한다.
- 동일한 연어는 문서당 한 번 집계한다. 출현 횟수와 문서 빈도를 구분한다.
- 실행: python3 analysis/scripts/12_agent_collocations.py.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parents[2]
SUFFIX = "2023_2026q3"
SRC = ROOT / "preprocessing/output" / f"corpus_llm_agent_all_{SUFFIX}.csv"
CACHE = ROOT / "analysis/output/cache"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = CACHE / f"agent_colloc_{SUFFIX}.jsonl"
VER = ROOT / "verification"

HEAD_RE = re.compile(r"^(?:(?P<prefix>[a-z0-9]+(?:-[a-z0-9]+)*)-)?agents?$")
MAX_SKIP = 2          # 머리 앞에서 한정사·소유 대명사를 건너뛰는 최대 단위 수


def units_of(doc):
    """토큰 → 단위. 붙임표로 공백 없이 이어진 토큰을 합친다. 각 단위: (form, content, adjacent_prev)"""
    out = []
    toks = list(doc)
    i = 0
    while i < len(toks):
        t = toks[i]
        parts = [t]
        # t '-' u 가 공백 없이 이어지면 합친다(여러 번)
        while (i + 2 < len(toks) and toks[i + 1].text == "-" and parts[-1].whitespace_ == "" and toks[i + 1].whitespace_ == ""
               and not toks[i + 2].is_punct and not toks[i + 2].is_space):
            parts.append(toks[i + 2]); i += 2
        if len(parts) > 1:
            form = re.sub(r"[^a-z0-9-]", "", "-".join(p.text for p in parts)).strip("-")   # (llm)-based → llm-based
            content = any(not p.is_stop and not p.is_punct for p in parts) and bool(re.search(r"[a-z]", form))
            kind = "compound"
        else:
            tag = t.tag_
            if tag in ("NN", "NNS", "NNP", "NNPS"):
                form, kind = t.lemma_, "noun"                       # agents → agent, systems → system
            elif tag in ("JJ", "JJR", "JJS", "VBG", "VBN"):
                form, kind = t.text, "adj"                           # autonomous, embodied, coding (표면형; 'coding'→'cod' 같은 표제어 오류 회피)
            else:
                form, kind = t.text, "other"                         # 동사·부사·전치사 등: 수식어가 아니므로 연어로 세지 않는다
            form = re.sub(r"[^a-z0-9-]", "", form).strip("-")   # 'llm)-based'처럼 괄호가 붙은 토큰 정리
            content = (kind != "other") and not (t.is_stop or t.is_punct or t.is_space or t.like_num) and bool(re.search(r"[a-z]", form))
        out.append((form, content, parts[0], parts[-1], kind if len(parts) > 1 else (kind if content else "other")))
        i += 1
    return out


def collocates(units):
    """머리 단위마다 (prefix, left1, left2, right1)과 agentic 뒤 말을 뽑는다."""
    res = {"prefix": [], "left1": [], "left2": [], "right1": [], "agentic_right1": [], "n_head": 0, "n_agentic": 0}
    n = len(units)
    for k, (form, content, first, last, kind) in enumerate(units):
        m = HEAD_RE.match(form)
        if form == "agentic":
            res["n_agentic"] += 1
            if k + 1 < n and units[k + 1][1] and units[k + 1][4] in ("noun", "compound"):
                res["agentic_right1"].append(units[k + 1][0])
            continue
        if not m:
            continue
        res["n_head"] += 1
        if m.group("prefix"):
            res["prefix"].append(m.group("prefix"))
        # 앞 말: 바로 앞 내용 단위. 한정사·소유 대명사(the, an, our, their)만 건너뛴다(최대 MAX_SKIP). 구두점·전치사·동사 등을 만나면 멈춘다
        li = None
        for j in range(k - 1, max(-1, k - 1 - MAX_SKIP - 1), -1):
            if units[j][1]:
                li = j; break
            if units[j][2].pos_ not in ("DET", "PRON"):
                break
        if li is not None:
            res["left1"].append(units[li][0])
            if li == k - 1 and li - 1 >= 0 and units[li - 1][1]:
                res["left2"].append(units[li - 1][0] + " " + units[li][0])
        # 뒤 말: 바로 다음 단위가 명사·합성어일 때만(agent framework, agent system). 'agents are', 'agents increasingly', 'agents.' 은 없음
        if k + 1 < n and units[k + 1][1] and units[k + 1][4] in ("noun", "compound"):
            res["right1"].append(units[k + 1][0])
    return res


def main() -> None:
    df = pd.read_csv(SRC, dtype={"id": str}, usecols=["id", "title", "abstract"])
    texts = (df.title.fillna("") + ". " + df.abstract.fillna("")).str.lower().tolist()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    n_with_head = 0; tot = Counter()
    with open(OUT, "w") as f:
        for pid, doc in zip(df.id, nlp.pipe(texts, batch_size=256, n_process=4)):
            r = collocates(units_of(doc))
            rec = {"id": pid, "n_head": r["n_head"], "n_agentic": r["n_agentic"]}
            for slot in ("prefix", "left1", "left2", "right1", "agentic_right1"):
                c = Counter(r[slot]); rec[slot] = dict(c)          # 단어 → 그 논문 안 출현 수(문서 빈도는 키의 존재로 센다)
            n_with_head += r["n_head"] > 0; tot["head"] += r["n_head"]; tot["agentic"] += r["n_agentic"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ver = {"generated": date.today().isoformat(), "script": "analysis/scripts/12_agent_collocations.py", "input": str(SRC.relative_to(ROOT)),
           "spacy": spacy.__version__, "model": "en_core_web_sm " + nlp.meta["version"], "n_docs": int(len(df)), "n_docs_with_agent_head": int(n_with_head),
           "n_head_occurrences": int(tot["head"]), "n_agentic_occurrences": int(tot["agentic"]),
           "rules": {"head": HEAD_RE.pattern, "max_skip_stop_punct": MAX_SKIP, "unit": "hyphen-joined tokens merged(surface, punctuation stripped); noun → lemma; adjective/participle(JJ*,VBG,VBN) → surface; other POS are not collocates", "left_skip": "DET/PRON only", "right": "noun or compound immediately after head",
                     "doc_frequency": "노트북에서 논문당 1로 센다"},
           "output": str(OUT.relative_to(ROOT))}
    p = VER / f"agent_colloc_extract_{date.today():%Y%m%d}.json"; p.write_text(json.dumps(ver, ensure_ascii=False, indent=1))
    print(json.dumps(ver, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
