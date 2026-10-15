#!/usr/bin/env python3
"""- 입력: verification의 자료·LDA·키워드군 정본.
- 출력: paper/tables의 표 1·2·3·4.
- 실행: python paper/scripts/05_make_paper_tables.py --lda
"""
from __future__ import annotations

import argparse

import glob

import json

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

LANG_TOT_KEY = "cjk_lang_by_lang_total"

from decimal import Decimal, ROUND_HALF_UP

def r1(x: float, nd: int = 1) -> str:
    """half-up 반올림 문자열(파이썬 기본 round는 은행가 반올림·부동소수 오차로 0.95→0.9가 된다)."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

def _latest(pattern: str) -> dict:
    return json.loads(Path(sorted(glob.glob(str(ROOT / pattern)))[-1]).read_text())

def make_lda_tables() -> None:
    """표 1(자료, 3.1절) · 표 2(LDA 토픽 25개와 언어 집합 관측(기대), 4.2~4.3절) · 표 4(키워드군, 4.4절)"""
    out_dir = ROOT / "paper/tables"; out_dir.mkdir(parents=True, exist_ok=True)
    ov = _latest("verification/overview_notebook_*.json"); ld = _latest("verification/topics_lda_*.json"); hn = _latest("verification/hclt_notebook_*.json")
    # 표 1 ─ 연도별 자료 규모
    t1 = ov["table1"]; rows = []
    for y, r in t1.items():
        name = {"2026 (Jan-Sep)": "2026 (1$\\sim$9월)", "total": "합계"}.get(y, y)
        rows.append(f"{name} & {r['cs papers (5 categories)']:,} & {r['LLM-agent papers']:,} & {r1(100 * r['LLM-agent papers'] / r['cs papers (5 categories)'], 2)} & {r['target Korean']} & {r['target Chinese']} & {r['target Japanese']} & {r['target CJK (any)']} \\\\")
    rows.insert(-1, "\\midrule")
    dd = ov["same_title_dedup"]
    tex1 = ("% 자동 생성: paper/scripts/05_make_paper_tables.py --lda (정본 overview_notebook_" + ov["generated"] + ".json)\n"
            "\\begin{table}[!htbp]\n\\centering\n\\caption{arXiv 연도별 자료 규모. 연도는 arXiv 식별자의 최초 제출 연월이며 2026년은 1$\\sim$9월이다. 수집 논문은 다섯 개 분류에서 중복을 제거한 편수, "
            "LLM 에이전트는 정의(\\ref{sec:method-data}절)에 따라 추출한 편수, 언어 열은 사람의 검수로 확정한 대상 언어 집합의 편수이다. 두 언어를 다룬 논문은 각 언어별 편수에 포함하되, 합계에서는 중복을 제거하였다.}\n\\label{tab:data}\n"
            "\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n연도 & 수집 논문 & \\shortstack{LLM\\\\에이전트} & 비중(\\%) & 한국어 & 중국어 & 일본어 & 합계 \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (out_dir / "tab_data_year.tex").write_text(tex1)
    # 표 2 ─ LDA 토픽 25개: 순위, 상위 단어, 전체 편수(연도별), 언어 집합 관측(기대)
    wt = sorted(ld["world_topic_table"].items(), key=lambda kv: kv[1]["world_rank"])
    N = ld["n_docs"]; sizes = {l: int(sum(r[f"{l}_obs"] for _, r in wt)) for l in ("ko", "zh", "ja")}
    def cell(r, l, highlight=False):
        exp = r["world_total"] / N * sizes[l]; obs = r[f"{l}_obs"]        # 기대 편수는 반올림 전 값으로 판정(재검산 7번)
        mark = ("$\\uparrow$" if obs > 1.5 * exp else "$\\downarrow$" if obs < 0.5 * exp else "") if exp >= 1 else ""
        shown = f"\\textbf{{{int(obs)}}}" if highlight else str(int(obs))
        return f"{shown} ({r1(exp)}; {r1(r[f'{l}_exp_lm'])}){mark}"   # 둘째 기대값 = 언어명 언급 논문 기준(4.1절)
    lines = []
    for t, r in wt:
        words = tex_escape(", ".join(ld["topics"][t]["top_words"][:5]))
        label = tuple(ld["topics"][t]["top_words"][:2])
        expansion = label in (("software", "execution"), ("retrieval", "graph"))
        ko_focus = expansion or label == ("training", "fine")
        zh_focus = label in (("retrieval", "graph"), ("reasoning", "multi"), ("evaluation", "benchmark"))
        ja_focus = label == ("social", "human")
        world = f"{int(r['world_total']):,}"
        if expansion:
            world = f"\\textbf{{{world}}}"
        yc = "/".join(f"{int(r[f'world_n_{y}']):,}" for y in (2023, 2024, 2025, 2026))
        lines.append(f"{int(r['world_rank'])} & {words} & {world} ({r1(100 * r['world_total'] / N)}) & {yc} & {cell(r, 'ko', ko_focus)} & {cell(r, 'zh', zh_focus)} & {cell(r, 'ja', ja_focus)} \\\\")
        if int(r["world_rank"]) in (8, 17):
            lines.append("\\midrule")
    tex2 = ("% 자동 생성: paper/scripts/05_make_paper_tables.py --lda (정본 topics_lda_" + ld["generated"] + ".json)\n"
            "\\begin{widetable}\n\\centering\n\\caption{LDA 토픽 " + str(ld["K"]) + f"개(arXiv {N:,}편, 전체 편수가 많은 순). 상위 단어는 각 토픽에서 단어 확률이 높은 순으로 5개의 단어를 추출한 값이다. 전체 열은 총 편수(비중 \\%)와 연도별 편수 2023/2024/2025/2026(1$\\sim$9월)이다. "
            "언어 열은 사람의 검수로 확정한 집합의 관측 편수(전체 집합 기준 기대 편수; 언어명 언급 논문 기준 기대 편수)이다. 기대 편수는 비교 집합의 토픽 비중에 해당 언어 집합의 논문 수를 곱하여 구하였다. "
            "$\\uparrow$는 관측 편수가 전체 집합 기준 기대 편수의 1.5배 초과, $\\downarrow$는 절반 미만(기대 편수가 1편 이상인 경우). 가로줄은 상위 8개·중위 9개·하위 8개의 경계이다. 화살표는 기술적 비교 표시이다.}\n\\label{tab:topics}\n"
            "\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{rlrrrrr}\n\\toprule\n"
            f"순위 & 상위 단어 & 전체 편수 (\\%) & 연도별 편수 & 한국어 ({sizes['ko']}) & 중국어 ({sizes['zh']}) & 일본어 ({sizes['ja']}) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n\\end{widetable}\n")
    (out_dir / "tab_lda_topics.tex").write_text(tex2)
    # 표 4 ─ 키워드군 비율: 전체 · 언어 집합 · HCLT
    fb = hn["family_by_set"]; n = fb["n"]; rows = []
    for f, r in fb["rows"].items():
        if f in ("agent", "llm"):
            continue
        rows.append(f"{tex_escape(f.replace('_', ' '))} & {r['arXiv world']:.1f} & {int(r['n Korean'])} ({r['Korean']:.0f}) & {int(r['n Chinese'])} ({r['Chinese']:.0f}) & {int(r['n Japanese'])} ({r['Japanese']:.0f}) & {int(r['n HCLT 2023-26'])} ({r['HCLT 2023-26']:.0f}) \\\\")
    tex4 = ("% 자동 생성: paper/scripts/05_make_paper_tables.py --lda (정본 hclt_notebook_" + hn["generated"] + ".json)\n"
            "\\begin{table}[!t]\n\\centering\n\\caption{키워드군별 논문 비율(\\%). 전체·언어별 열은 arXiv 자료이다. 전체 열은 비율, 나머지 열은 편수(비율)이다.}\n\\label{tab:families}\n\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
            f"키워드군 & \\shortstack{{전체\\\\({n['arXiv world']:,})}} & \\shortstack{{한국어\\\\({n['Korean']})}} & \\shortstack{{중국어\\\\({n['Chinese']})}} & \\shortstack{{일본어\\\\({n['Japanese']})}} & \\shortstack{{HCLT\\\\논문집\\\\({n['HCLT 2023-26']})}} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (out_dir / "tab_families.tex").write_text(tex4)
    print("자료·LDA·키워드군 표 생성(--lda):", "tab_data_year.tex, tab_lda_topics.tex, tab_families.tex")


def make_collocation_summary() -> None:
    """- 저장된 연어 비율에서 네 가지 표현의 연도별 비교 표를 생성한다."""
    source = ROOT / "verification/agent_colloc_20261015.json"
    data = json.loads(source.read_text())
    entries = [
        ("수행하는 작업", "before", "coding", "coding agent", "코딩처럼 수행 목적을 명시한 에이전트 연구의 확대"),
        ("작업을 연결하는 절차", "after", "workflow", "agent workflow", "여러 단계의 작업을 어떻게 구성하고 연결할지에 관한 관심"),
        ("작업 중 정보를 유지하는 방식", "after", "memory", "agent memory", "기억을 에이전트의 구성 요소로 다루는 연구의 확대"),
        ("수행 결과를 비교하는 기준", "after", "benchmark", "agent benchmark", "에이전트의 수행 능력을 평가하는 연구의 확대"),
    ]
    rows = []
    for subject, side, word, phrase, meaning in entries:
        row = data["world"][side][word]
        change = f"{r1(row['pct_2023'])}\\% $\\rightarrow$ {r1(row['pct_2026'])}\\%"
        rows.append(f"{subject} & {phrase} & {change} & {meaning} " + r"\\")
    tex = (
        "% 자동 생성: verification/agent_colloc_20261015.json\n"
        "\\begin{widetable}\n\\centering\n"
        "\\caption{주요 연어의 비중 변화와 연구 대상의 구체화. 비율의 분모는 각 연도의 전체 집합이며, 2026년은 1$\\sim$9월이다.}\n"
        "\\label{tab:colloc-direction}\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.20\\textwidth}>{\\raggedright\\arraybackslash}p{0.20\\textwidth}>{\\centering\\arraybackslash}p{0.18\\textwidth}>{\\raggedright\\arraybackslash}p{0.36\\textwidth}@{}}\n"
        "\\toprule\n구체화되는 연구 대상 & 연어 & 2023년 $\\rightarrow$ 2026년 & 읽을 수 있는 변화 \\\\\n\\midrule\n"
        + "\n\\addlinespace[3pt]\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{widetable}\n"
    )
    (ROOT / "paper/tables/tab_colloc_direction.tex").write_text(tex)
    print("연어 변화 표 생성: tab_colloc_direction.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lda", action="store_true", help="표 1·2·3·4 생성")
    parser.parse_args()
    make_lda_tables()
    make_collocation_summary()
