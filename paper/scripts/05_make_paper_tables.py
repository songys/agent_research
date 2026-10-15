#!/usr/bin/env python3
"""- 입력: verification의 자료·LDA·키워드군 정본.
- 출력: paper/tables의 표 1·2·3.
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
            "\\begin{table}[!htbp]\n\\centering\n\\caption{arXiv 연도별 자료 규모. 연도는 arXiv 식별자의 최초 제출 연월이며 2026년은 1$\\sim$9월이다. 전체는 다섯 개 분류에서 중복을 제거한 논문 수, "
            "LLM 에이전트는 정의(\\ref{sec:method-data}절)에 따라 추출한 편수, 언어 열은 사람의 검수로 확정한 대상 언어 집합의 편수이다. 논문 한 편이 두 개의 대상 언어를 함께 다루면 각 언어 열에 한 편씩 집계하고, 합집합에서는 한 편으로 센다.}\n\\label{tab:data}\n"
            "\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n연도 & 전체 & \\shortstack{LLM\\\\에이전트} & 비중(\\%) & 한국어 & 중국어 & 일본어 & 합집합 \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (out_dir / "tab_data_year.tex").write_text(tex1)
    # 표 2 ─ LDA 토픽 25개: 순위, 상위 단어, 세계 편수(연도별), 기울기, 언어 집합 관측(기대)
    wt = sorted(ld["world_topic_table"].items(), key=lambda kv: kv[1]["world_rank"])
    N = ld["n_docs"]; sizes = {l: int(sum(r[f"{l}_obs"] for _, r in wt)) for l in ("ko", "zh", "ja")}
    def cell(r, l, highlight=False):
        exp = r["world_total"] / N * sizes[l]; obs = r[f"{l}_obs"]        # 기대 편수는 반올림 전 값으로 판정(재검산 7번)
        mark = ("$\\uparrow$" if obs > 1.5 * exp else "$\\downarrow$" if obs < 0.5 * exp else "") if exp >= 1 else ""
        shown = f"\\textbf{{{int(obs)}}}" if highlight else str(int(obs))
        return f"{shown} ({r1(exp)}; {r1(r[f'{l}_exp_lm'])}){mark}"   # 둘째 기대값 = 언어명을 언급한 세계 논문 기준(4.1절)
    lines = []
    for t, r in wt:
        words = tex_escape(", ".join(ld["topics"][t]["top_words"][:5]))
        label = tuple(ld["topics"][t]["top_words"][:2])
        expansion = label in (("software", "execution"), ("retrieval", "graph"))
        ko_focus = expansion or label == ("training", "fine")
        zh_focus = label == ("retrieval", "graph")
        world = f"{int(r['world_total']):,}"
        if expansion:
            world = f"\\textbf{{{world}}}"
        yc = "/".join(f"{int(r[f'world_n_{y}']):,}" for y in (2023, 2024, 2025, 2026))
        sl = r1(r["world_slope_pp_per_year"]); sl = sl if sl.startswith("-") else "+" + sl
        lines.append(f"{int(r['world_rank'])} & {words} & {world} ({r1(100 * r['world_total'] / N)}) & {yc} & {sl} & {cell(r, 'ko', ko_focus)} & {cell(r, 'zh', zh_focus)} & {cell(r, 'ja')} \\\\")
        if int(r["world_rank"]) in (8, 17):
            lines.append("\\midrule")
    tex2 = ("% 자동 생성: paper/scripts/05_make_paper_tables.py --lda (정본 topics_lda_" + ld["generated"] + ".json)\n"
            "\\begin{widetable}\n\\centering\n\\caption{LDA 토픽 " + str(ld["K"]) + f"개(arXiv {N:,}편, 전체 편수가 많은 순). 상위 단어는 각 토픽에서 단어 확률이 높은 다섯 개의 단어이다. 세계 열은 총 편수(비중 \\%)와 연도별 편수 2023/2024/2025/2026(1$\\sim$9월), "
            "기울기는 연도별 점유율의 직선 기울기(pp/년)이다. 언어 열은 사람의 검수로 확정한 집합의 관측 편수(세계 전체 기준 기대 편수; 언어명을 언급한 세계 논문 기준 기대 편수)이며 기대 = 토픽 비중 $\\times$ 집합 크기. "
            "$\\uparrow$는 관측 편수가 세계 전체 기준 기대 편수의 1.5배 초과, $\\downarrow$는 절반 미만(기대 1편 미만은 표시하지 않음). 가로줄은 상위 8개·중위 9개·하위 8개의 경계이다. 굵은 수치는 한국어 연구 방향의 비교 근거이다.}\n\\label{tab:topics}\n"
            "\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{rlrrrrrr}\n\\toprule\n"
            f"순위 & 상위 단어 & 세계 편수 (\\%) & 연도별 편수 & 기울기 & 한국어 ({sizes['ko']}) & 중국어 ({sizes['zh']}) & 일본어 ({sizes['ja']}) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n\\end{widetable}\n")
    (out_dir / "tab_lda_topics.tex").write_text(tex2)
    # 표 4 ─ 키워드군 비율: 세계 · 언어 집합 · HCLT
    fb = hn["family_by_set"]; n = fb["n"]; rows = []
    for f, r in fb["rows"].items():
        if f in ("agent", "llm"):
            continue
        rows.append(f"{tex_escape(f.replace('_', ' '))} & {r['arXiv world']:.1f} & {int(r['n Korean'])} ({r['Korean']:.0f}) & {int(r['n Chinese'])} ({r['Chinese']:.0f}) & {int(r['n Japanese'])} ({r['Japanese']:.0f}) & {int(r['n HCLT 2023-26'])} ({r['HCLT 2023-26']:.0f}) \\\\")
    tex4 = ("% 자동 생성: paper/scripts/05_make_paper_tables.py --lda (정본 hclt_notebook_" + hn["generated"] + ".json)\n"
            "\\begin{table}[H]\n\\centering\n\\caption{키워드군별 논문 비율(\\%). 세계·언어별 열은 arXiv 자료이다. 세계 열은 비율, 나머지 열은 편수(비율)이다. 용어 목록과 판정 기준은 \\ref{sec:method-families}절에 설명하였다.}\n\\label{tab:families}\n\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\columnwidth}{!}{\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
            f"키워드군 & \\shortstack{{세계\\\\({n['arXiv world']:,})}} & \\shortstack{{한국어\\\\({n['Korean']})}} & \\shortstack{{중국어\\\\({n['Chinese']})}} & \\shortstack{{일본어\\\\({n['Japanese']})}} & \\shortstack{{HCLT\\\\논문집\\\\({n['HCLT 2023-26']})}} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (out_dir / "tab_families.tex").write_text(tex4)
    print("표 1·2·3 생성(--lda):", "tab_data_year.tex, tab_lda_topics.tex, tab_families.tex")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lda", action="store_true", help="표 1·2·3 생성")
    parser.parse_args()
    make_lda_tables()
