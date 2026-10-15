#!/usr/bin/env python3
"""현재 원고의 분석 노트북을 생성한다.

입력: preprocessing/output/*_2023_2026q3.csv, verification/*.json, 분석 캐시.
출력: analysis/notebooks/*.ipynb(지정 번호만 생성; 인자가 없으면 6개 전체).
현재 원고: 01(3장 표 1), 02(4장 표 2·그림 1), 05(4.2.2 표 3),
07·08(4.1.1 본문), 10(4.2.1 연도 조정 회귀).
전체 20,107편; 대상 언어 합집합 226편. 국가 비교·BERTopic은 이전 분석이다.

현재 원고 재현은 paper_code/run.py와 REPRODUCIBILITY.md를 따른다.
이 생성기는 기존 노트북과 실행 출력을 덮어쓰므로 일반 재현 때 실행하지 않는다.
수정 후 특정 노트북을 만들 때만 격리 사본에서 번호를 지정한다:
python3 analysis/scripts/02_make_notebooks.py 01 02 05 07 08 10
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/notebooks"
OUT.mkdir(exist_ok=True)


def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


COMMON_SETUP = '''import glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "verification").exists())   # 노트북은 analysis/notebooks/ 에 있다
DATA = ROOT / "preprocessing/output"                                   # 입력: 전처리 산출(정제 CSV)
VER = ROOT / "verification"                                            # 수치의 정본 JSON
FIG = ROOT / "analysis/output/figures"; FIG.mkdir(parents=True, exist_ok=True)   # 출력: 노트북 그림. 원고 그림은 paper/figures/
TAB = ROOT / "analysis/output/tables"; TAB.mkdir(parents=True, exist_ok=True)    # 출력: 노트북 표
CACHE = ROOT / "analysis/output/cache"; CACHE.mkdir(parents=True, exist_ok=True) # 출력: 재실행용 캐시(git 제외)
TODAY = pd.Timestamp.today().strftime("%Y%m%d")
SUFFIX = "2023_2026q3"   # D30: 분석 기간 2023-01 ~ 2026-09 말뭉치

try:
    sys.path.insert(0, str(ROOT / "analysis/scripts"))
    import figure_style; figure_style.apply_style(); C = figure_style.COLORS; CYC = figure_style.COLOR_CYCLE
except Exception:
    C = {"green": "#46B07B", "orange": "#FF9E02", "teal": "#3A9E8F", "coral": "#E87B4A", "green_dark": "#2D7A53", "orange_dark": "#C47800", "green_light": "#8DD4AA", "orange_light": "#FFD080", "green_pale": "#C8EBDA", "orange_pale": "#FFECC0", "gray": "#7F8C8D", "gray_light": "#BDC3C7", "gray_dark": "#4A4A4A", "text": "#333333"}; CYC = [C["green"], C["orange"], C["teal"], C["coral"]]
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

LANGS = ["ko", "zh", "ja"]
LANG_NAME = {"ko": "Korean", "zh": "Chinese", "ja": "Japanese"}
YEARS = [2023, 2024, 2025, 2026]          # 2026은 1~9월(D30)
MIN_N = 10   # 투영 집합의 토픽별 건수가 이보다 작으면 lift를 보고하지 않는다

allc = pd.read_csv(DATA / f"corpus_llm_agent_all_{SUFFIX}.csv", dtype={"id": str})
cjk_signal = pd.read_csv(DATA / f"corpus_llm_agent_cjk_lang_{SUFFIX}.csv", dtype={"id": str})   # 키워드 신호 257편 + 사람 라벨(D36)
counts = json.loads(Path(sorted(glob.glob(str(VER / "corpus_counts_*.json")))[-1]).read_text())
YEAR_BASIS = counts["definitions"]["year_basis"]   # D31: "id" = arXiv id 접두(최초 제출 월)
LABEL_TARGET = "그 언어의 텍스트를 실제 연구 대상으로 다룸"; LABEL_MULTI = "다국어 실험의 한 언어로 포함"
cjk = cjk_signal[cjk_signal.cjk_lang].copy()       # 최종 대상 언어 집합 = 신호 ∧ 라벨 '실제 연구 대상'(D36)

# 정본 대조: 노트북이 읽은 자료가 정본 JSON과 같은지 확인한다
assert YEAR_BASIS == "id", "연도 기준이 D31(id 접두)이 아니다"
assert len(allc) == counts["pipeline"]["llm_agent_all"], "전체 건수가 정본과 다르다"
assert len(cjk_signal) == counts["pipeline"]["llm_agent_cjk_signal"], "대상 언어 신호 건수가 정본과 다르다"
assert len(cjk) == counts["pipeline"]["llm_agent_cjk_lang"], "대상 언어 집합 건수가 정본과 다르다"
assert (cjk_signal.lang_label != "").all(), "라벨 없는 신호 논문이 있다"
assert allc.groupby("year").size().to_dict() == {int(k): v for k, v in counts["llm_agent_by_year"].items()}
assert set(cjk.id) <= set(allc.id) and allc.id.is_unique
for l in LANGS:
    allc[f"lang_{l}"] = allc[f"lang_{l}"].astype(bool); allc[f"lang_{l}_signal"] = allc[f"lang_{l}_signal"].astype(bool)
assert {l: int(allc[f"lang_{l}"].sum()) for l in LANGS} == counts["cjk_lang_by_lang_total"]
N_ALL_BY_YEAR = {int(k): v for k, v in counts["pipeline"]["unique_by_year"].items()}   # 분모: 다섯 cs 분류의 고유 논문 수
print("정본 대조 통과: LLM 에이전트", len(allc), "| 한중일어 신호", len(cjk_signal), "| 대상 언어 집합(사람 라벨)", len(cjk), "| 연도 기준", YEAR_BASIS, "| 기간", counts["definitions"]["period"][:24])'''



# 투영 공통 함수: 토픽 열 이름을 받아 국가·언어 집합의 분포·lift·순열 검정을 계산


# ─────────────────────────────────────────────────────────────────────────────
# 01 corpus overview
# ─────────────────────────────────────────────────────────────────────────────
nb01 = nb([
    ("md", """# 01. 말뭉치 개요 — LLM 에이전트 논문 전체와 한중일어 대상 집합 (2023-01 ~ 2026-09)

- **목적**: 토픽 학습 집합(세계 전체 = 수집한 다섯 cs 분류의 LLM 에이전트 논문)과 투영 집합(한국어·중국어·일본어 대상)의 규모·분포를 확인하고 논문 3장의 자료 표와 그림 1을 만든다. 방법 절에 적을 증빙(연도 기준, 중복 제거 편수)도 표로 남긴다.
- **입력**: `preprocessing/output/corpus_llm_agent_all_2023_2026q3.csv`, `corpus_llm_agent_cjk_lang_2023_2026q3.csv`, 정본 `verification/corpus_counts_*.json`(01_build_corpus.py)
- **정의** (DECISIONS)
  - LLM 에이전트 논문(D1): 제목·초록에 agent 계열 단어와 LLM·생성형 AI 계열 단어가 함께 나오는 논문
  - 대상 언어(D10·D36): 제목·초록 키워드로 잡은 korean / chinese / japanese 신호(257편)에 사람 검수 라벨을 붙였다. 집합에 넣는 것은 "그 언어의 텍스트를 실제 연구 대상으로 다룸"뿐이다. 영어와 그 언어만 쓴 논문은 실제 연구 대상, 영어·그 언어 외 다른 언어가 더 있으면 "다국어 실험의 한 언어로 포함"(집합 밖, 따로 센다). 라벨은 논문 단위라 복수 태그 논문은 모든 태그에 같은 라벨이다. 논문이 다루는 언어이며 쓰인 언어가 아니다
  - 연도(D31): arXiv id 접두의 최초 제출 월(YYMM). 수집본의 published는 최신 판 날짜라 쓰지 않는다. 기간은 2023-01~2026-09(D30), 2026은 9개월치
  - 소속 국가 축은 쓰지 않는다(D30). 2026-03~09 수집본에는 소속이 없다
- **주의**
  - 대상 언어 태그는 한 논문에 여러 개가 붙을 수 있어 언어 합이 집합 크기를 넘을 수 있다
  - 비율의 분모 "전체 cs 논문"은 다섯 분류(cs.AI/CL/CV/CY/LG)의 고유 논문 수다(정본 `pipeline.unique_by_year`)
"""),
    ("code", COMMON_SETUP),
    ("md", """## 1. 자료 표 — 연도별 전체 cs 논문·LLM 에이전트·한중일어 대상

- 계산: 연도별 다섯 분류 고유 논문 수, LLM 에이전트 논문 수와 비중(%), 그중 한국어·중국어·일본어 대상 논문 수. 합계 행을 둔다
- 출력: `analysis/output/tables/table1_year_overview.csv` (논문 3장 자료 표)
"""),
    ("code", '''t1 = pd.DataFrame({"cs papers (5 categories)": pd.Series(N_ALL_BY_YEAR), "LLM-agent papers": allc.groupby("year").size()}).reindex(YEARS).fillna(0).astype(int)
t1["LLM-agent share %"] = (100 * t1["LLM-agent papers"] / t1["cs papers (5 categories)"]).round(2)
for l in LANGS:
    t1[f"target {LANG_NAME[l]}"] = allc[allc[f"lang_{l}"]].groupby("year").size().reindex(YEARS).fillna(0).astype(int)
t1["target CJK (any)"] = cjk.groupby("year").size().reindex(YEARS).fillna(0).astype(int)
tot = t1.sum(numeric_only=True); tot["LLM-agent share %"] = round(100 * t1["LLM-agent papers"].sum() / t1["cs papers (5 categories)"].sum(), 2)
t1.loc["total"] = tot; t1.index = [str(i) if i != 2026 else "2026 (Jan-Sep)" for i in t1.index]
t1.to_csv(TAB / "table1_year_overview.csv"); t1'''),
    ("md", """## 1b. 대상 언어 사람 검수 라벨 — 신호 257편의 라벨 분포 (D36)

- 계산: 키워드 신호(target_langs_signal)별 라벨 분포. 집합에 들어가는 것은 "그 언어의 텍스트를 실제 연구 대상으로 다룸"뿐. "다국어 실험의 한 언어로 포함"은 언어별로 따로 센다
- 입력: `cjk_signal`(257편, 라벨 열 lang_label), 정본 `language_label_review`(검수 파일 경로·재점검 변경 수)
- 출력: `analysis/output/tables/table1b_lang_label_review.csv` (방법 절의 검수 결과 표)
- 주의: 라벨은 논문 단위. 신호가 둘 이상인 논문(예 ko;ja)은 모든 태그에 같은 라벨. 에이전트 무관 분모(3절)는 검수하지 않았으므로 키워드 신호 그대로
"""),
    ("code", '''lr = counts["language_label_review"]
print(f"검수 파일: {lr['main_file']} ({lr['n_main']}편) + 재점검 {lr['recheck_file']} ({lr['n_recheck']}편, 라벨 바뀜 {lr['n_recheck_changed']}편)")
t1b = pd.crosstab(cjk_signal.target_langs_signal, cjk_signal.lang_label.replace("", "(없음)"), margins=True, margins_name="total")
assert int(t1b.loc["total", LABEL_TARGET]) == len(cjk)
t1b.to_csv(TAB / "table1b_lang_label_review.csv")
per_lang = pd.DataFrame({LANG_NAME[l]: {"signal": int(cjk_signal[f"lang_{l}_signal"].sum()), "target (in set)": int(cjk_signal[f"lang_{l}"].sum()),
                                        "multilingual (out)": int((cjk_signal[f"lang_{l}_signal"] & cjk_signal.lang_multilingual_human).sum()),
                                        "other (out)": int((cjk_signal[f"lang_{l}_signal"] & ~cjk_signal.lang_target_human & ~cjk_signal.lang_multilingual_human).sum())} for l in LANGS}).T
assert per_lang["target (in set)"].to_dict() == {LANG_NAME[l]: counts["cjk_lang_by_lang_total"][l] for l in LANGS}
print(per_lang.to_string()); t1b'''),
    ("md", """## 1c. 민감도 — 라벨 규칙을 뒤집으면, 정의 밖에 남는 에이전트 논문은 (리뷰 2026-10-14)

- (가) 대상 언어 라벨 규칙 "영어와 그 언어만이면 실제 연구 대상"을 뒤집어 "영어가 언급되면 다국어 실험"으로 보면 집합이 얼마나 줄어드는가. 검수 입력(제목·초록)에서 'english'가 있는 실제-대상 논문을 센다
- (나) agent 계열 단어는 있고 LLM 계열 용어는 없어 정의(D1) 밖에 남은 논문 가운데 agentic, ai agent, coding/web/gui/computer-use agent, tool calling, mcp 표현이 있는 논문 수(연도별). 소문자 부분 문자열 일치로 센 어림값이며 정의를 바꾸지 않는다
- 출력: 정본 `sensitivity` 항목
"""),
    ("code", '''rv = cjk_signal.set_index("id")[["title", "abstract"]].to_dict(orient="index")
tgt = cjk_signal[cjk_signal.cjk_lang]
has_en = tgt.id.map(lambda i: "english" in (rv[i]["title"] + " " + rv[i]["abstract"]).lower())
rev_rule = {l: int((tgt[f"lang_{l}"] & ~has_en).sum()) for l in LANGS}; rev_rule["any"] = int((~has_en).sum())
print("규칙을 뒤집으면 남는 집합:", rev_rule, "| 현재", {l: int(tgt[f"lang_{l}"].sum()) for l in LANGS}, len(tgt))
tag = pd.read_csv(DATA / f"corpus_agent_tag_{SUFFIX}.csv", dtype={"id": str}, usecols=["id", "year", "title", "abstract", "is_llm_agent"])
out_def = tag[~tag.is_llm_agent.astype(bool)]; tx = (out_def.title.fillna("") + " " + out_def.abstract.fillna("")).str.lower()
CUES = ["agentic", "ai agent", "coding agent", "web agent", "gui agent", "computer-use agent", "computer use agent", "tool calling", "tool-calling", "function calling", " mcp"]
cue = tx.map(lambda x: any(c in x for c in CUES))
outside = out_def[cue].groupby("year").size().reindex(YEARS, fill_value=0)
print(f"agent 태그만 있고 정의 밖인 논문 {len(out_def):,}편 중 LLM 에이전트 단서 보유 {int(cue.sum()):,}편:", outside.to_dict())
sensitivity = {"label_rule_reversed_set": rev_rule, "label_rule_note": "'english'가 제목·초록에 있는 실제-대상 논문을 다국어로 간주했을 때 남는 편수",
               "agent_tag_outside_definition": int(len(out_def)), "outside_with_llm_agent_cues": {str(k): int(v) for k, v in outside.items()}, "cues": CUES}'''),
    ("md", """## 2. 방법 절 증빙 — 연도 기준과 중복 제거

- 계산: (가) 같은 집합을 published(수집본 날짜) 기준으로 셌을 때의 연도별 고유 논문 수를 id 접두 기준과 나란히 둔다. (나) 제목이 같은 다른 id(재제출·중복 등록)를 한 편으로 합친 건수.
- 입력: 정본 `pipeline.unique_by_year`, `pipeline.unique_by_year_other_basis`, `pipeline.same_title_different_id`
- 출력: `analysis/output/tables/table0_year_basis_and_dedup.csv`
"""),
    ("code", '''pl = counts["pipeline"]
t0 = pd.DataFrame({"id prefix (first submission month, D31)": {int(k): v for k, v in pl["unique_by_year"].items()},
                   "published (harvest-time latest version)": {int(k): v for k, v in pl["unique_by_year_other_basis"].items()}}).sort_index().fillna(0).astype(int)
t0.to_csv(TAB / "table0_year_basis_and_dedup.csv")
dd = pl["same_title_different_id"]
print(f"같은 제목·다른 id: {dd['groups']}묶음 {dd['rows']}편 → {dd['removed']}편 제거 (규칙: {dd['rule']})")
print(f"교차 등재 포함 원 행 수 {pl['rows_raw_all_files']:,} → 고유 id {pl['unique_ids_all_years']:,} → 기간 안 {pl['unique_ids_in_period']:,}")
t0'''),
    ("md", """## 3. 대상 언어 논문 안에서 에이전트의 비중

- 계산: 그 언어를 대상으로 한 논문 전체(에이전트 무관, 정본 `language_target_papers_all_topics`) 대비 LLM 에이전트 논문의 비율
- 해석: "한국어를 다루는 논문 중 에이전트 논문은 몇 %인가". 언어권 연구 안에서 에이전트 주제의 비중
"""),
    ("code", '''lt = counts["language_target_papers_all_topics"]
t3 = pd.DataFrame({LANG_NAME[l]: {"target-language papers (all topics)": lt[l]["total"], "LLM-agent among them": lt[l]["llm_agent"],
                                   "share %": round(100 * lt[l]["llm_agent"] / lt[l]["total"], 2)} for l in LANGS}).T
t3.to_csv(TAB / "table3_agent_share_within_language.csv"); t3'''),
    ("md", """## 4. 그림 1 — 연도별 건수

- 왼쪽: 전체 LLM 에이전트 논문(막대)과 전체 cs 논문 대비 비중(선)
- 오른쪽: 대상 언어별 LLM 에이전트 논문 수(로그 축). 중국어와 한국어·일본어의 규모 차이를 한 축에 둔다
- 출력: `analysis/output/figures/fig1_year_counts.png`
"""),
    ("code", '''fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.15]}); xl = [str(y) if y != 2026 else "2026\\n(Jan–Sep)" for y in YEARS]
ax = axes[0]; tot = allc.groupby("year").size().reindex(YEARS)
ax.bar(range(4), tot.values, color=C["green"], width=0.62, label="LLM-agent papers", zorder=3)
for i, v in enumerate(tot.values): ax.text(i + 0.08, v + 0.02 * tot.max(), f"{v:,}", ha="left", va="bottom", fontsize=8, color=C["text"])
ax2 = ax.twinx(); share = 100 * tot / pd.Series(N_ALL_BY_YEAR).reindex(YEARS)
ax2.plot(range(4), share.values, marker="o", color=C["orange"], lw=1.8, label="share of cs papers (%)", zorder=4); ax2.set_ylim(0, max(14, share.max() + 1)); ax2.set_ylabel("share (%)"); ax2.grid(False); ax2.spines["top"].set_visible(False)
for i, v in enumerate(share.values): ax2.text(i - 0.1, v + 0.35, f"{v:.1f}%", fontsize=7.5, color=C["orange_dark"], ha="right", va="bottom")
ax.set_xticks(range(4)); ax.set_xticklabels(xl); ax.set_ylabel("papers"); ax.set_title("LLM-agent papers in five cs categories"); ax.set_ylim(0, tot.max() * 1.15)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left")
ax = axes[1]; w = 0.26; cols = [C["orange"], C["green"], C["teal"]]
for i, l in enumerate(LANGS):
    vals = allc[allc[f"lang_{l}"]].groupby("year").size().reindex(YEARS).fillna(0).astype(int)
    bars = ax.bar(np.arange(4) + (i - 1) * w, vals.values, width=w, label=LANG_NAME[l], color=cols[i], zorder=3)
    for b, v in zip(bars, vals.values): ax.text(b.get_x() + b.get_width() / 2, max(v, 0.8) * 1.08, str(v), ha="center", va="bottom", fontsize=7, color=C["text"])
ax.set_yscale("log"); ax.set_ylim(0.7, 200); ax.set_xticks(range(4)); ax.set_xticklabels(xl); ax.set_ylabel("papers (log scale)"); ax.set_title("Target-language LLM-agent papers (human-labeled)"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig1_year_counts.png", bbox_inches="tight", dpi=200); plt.show()'''),
    ("md", """## 5. 정본 저장과 검수 목록

- 저장: 표 1·1b·0·3의 값을 `verification/overview_notebook_YYYYMMDD.json`에 남긴다(노트북 산출 정본). 한중일어 대상 논문 목록은 검수 입력 참고용 CSV
"""),
    ("code", '''rev = cjk[["id", "year", "month", "target_langs", "primary_category", "title"]].sort_values(["target_langs", "year", "id"])
rev["url"] = "https://arxiv.org/abs/" + rev.id; rev.to_csv(TAB / "cjk_lang_papers_for_review.csv", index=False)
out = {"generated": TODAY, "notebook": "analysis/notebooks/01_corpus_overview.ipynb", "period": counts["definitions"]["period"], "year_basis": YEAR_BASIS,
       "table1": {str(k): {c: (float(v) if "share" in c else int(v)) for c, v in r.items()} for k, r in t1.iterrows()},
       "table0_year_basis": {str(k): {c: int(v) for c, v in r.items()} for k, r in t0.iterrows()}, "same_title_dedup": dd,
       "sensitivity": sensitivity,
       "table1b_lang_label_review": {"per_lang": {str(k): {c: int(v) for c, v in r.items()} for k, r in per_lang.iterrows()}, "label_by_signal": {str(k): {str(c): int(v) for c, v in r.items()} for k, r in t1b.iterrows()}, "review_files": [lr["main_file"], lr["recheck_file"]]},
       "table3": {str(k): {c: float(v) for c, v in r.items()} for k, r in t3.iterrows()}}
p = VER / f"overview_notebook_{TODAY}.json"; p.write_text(json.dumps(out, ensure_ascii=False, indent=1)); print("saved", p, "| 검수 목록", len(rev), "건")'''),
])

# ─────────────────────────────────────────────────────────────────────────────
# 02 LDA
# ─────────────────────────────────────────────────────────────────────────────
nb02 = nb([
    ("md", """# 02. LDA 토픽 모델 — 세계 전체(LLM 에이전트 논문)에서 학습, 한중일어 대상 집합 투영 (2023-01 ~ 2026-09)

- **목적**: 확률 토픽 모델 LDA(Latent Dirichlet Allocation)로 LLM 에이전트 연구의 주제 구조를 뽑고, 그 위에 한국어·중국어·일본어 대상 논문을 투영해 세계 전체 분포와 비교한다(논문 4장). D30에 따라 BERTopic 대신 LDA가 주 모델이다.
- **참고 코드**: pytextbook 7.1 `seoul-120-LDA.ipynb` — `CountVectorizer` → `sklearn.decomposition.LatentDirichletAllocation(random_state=42)` → `pyLDAvis`. 이 노트북은 그 틀을 따르고 다음을 더한다
  - 토큰: spaCy `en_core_web_sm` 표제어(lemma, D26). 영어 불용어 + 도메인 불용어 제거, 세 글자 이상 알파벳 토큰
  - 토픽 수 K: 참고 코드는 분류 수(10)를 썼으나 이 자료에는 정답 분류가 없어 K ∈ {10, 15, 20, 25, 30}을 c_v coherence(gensim)와 보류 집합 perplexity로 비교해 고른다
  - 문서 배정: 문서-토픽 확률의 최대 토픽(argmax). 토픽 라벨은 상위 두 단어(모델 출력, D20)
- **입력**: `preprocessing/output/corpus_llm_agent_all_2023_2026q3.csv`의 제목+초록. 표제어 캐시 `analysis/output/cache/lemmas_2023_2026q3.jsonl`
- **출력**: `verification/topics_lda_YYYYMMDD.json`(정본), `analysis/output/tables/lda_*.csv`, `analysis/output/figures/fig2*_lda_*.png`, `lda_pyldavis_K*.html`
- **주의**
  - 투영 집합의 토픽별 건수가 10건 미만이면 lift를 보고하지 않고 "관측 n편 / 기대 m편"으로만 적는다
  - 대상 언어 라벨은 사람 검수 전이다. 검수 뒤 라벨이 바뀌면 이 노트북을 다시 실행한다
"""),
    ("code", COMMON_SETUP),
    ("md", """## 1. 전처리 — spaCy 표제어 토큰

- 계산: 제목+초록을 소문자로 바꿔 spaCy로 표제어화(구문 분석·개체명 끔). 알파벳 세 글자 이상, 영어 불용어·도메인 불용어 제외
- 캐시: id → 토큰 목록을 jsonl로 저장해 재실행 때 다시 쓴다(자료가 바뀌면 캐시의 id 집합이 달라져 자동 재계산)
"""),
    ("code", '''from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
DOMAIN_STOP = {"paper", "propose", "proposed", "method", "methods", "result", "results", "show", "shows", "model", "models", "based",
               "approach", "approaches", "task", "tasks", "performance", "llm", "llms", "agent", "agents", "large", "language", "framework",
               "novel", "existing", "demonstrate", "experiments", "experimental", "study", "work", "using", "use", "data", "code", "available",
               "https", "github", "com", "et", "al", "paper", "present", "significantly", "propose", "introduce", "achieve", "outperform", "sota", "state", "art",
               "datum", "cod"}   # spaCy 표제어 특이값: data→datum, code(s)→cod 는 도메인 불용어(data, code)와 같은 말이다
STOP = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP
cache = CACHE / f"lemmas_{SUFFIX}.jsonl"
lemmas = {}
if cache.exists():
    for line in cache.open():
        d = json.loads(line); lemmas[d["id"]] = d["t"]
need = [i for i in allc.id if i not in lemmas]
if need:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts = (allc.set_index("id").loc[need].title.fillna("") + ". " + allc.set_index("id").loc[need].abstract.fillna("")).str.lower().tolist()
    with cache.open("a") as f:
        for i, doc in zip(need, nlp.pipe(texts, batch_size=128)):
            toks = [t.lemma_ for t in doc if t.is_alpha and len(t.lemma_) > 2 and not t.is_stop]   # 불용어는 캐시 뒤 단계에서 거른다(사전이 바뀌어도 캐시 재사용)
            lemmas[i] = toks; f.write(json.dumps({"id": i, "t": toks}) + "\\n")
tokens = [[w for w in lemmas[i] if w not in STOP] for i in allc.id]   # 캐시는 걸러지기 전 상태일 수 있으므로 불용어를 여기서 다시 거른다
assert len(tokens) == len(allc) and len(set(allc.id) - set(lemmas)) == 0
print("문서", len(tokens), "건 | 평균 토큰", round(np.mean([len(t) for t in tokens]), 1), "| 캐시", cache.name, "(새로 계산", len(need), "건)")'''),
    ("md", """## 2. 문서-단어 행렬

- 계산: 참고 코드와 같은 `CountVectorizer`. 토큰은 이미 표제어이므로 공백 분리만 한다. 문서 빈도 10 이상·50% 이하 단어만
"""),
    ("code", '''from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False, token_pattern=None, min_df=10, max_df=0.5)
dtm = cv.fit_transform([" ".join(t) for t in tokens]); vocab = np.array(cv.get_feature_names_out())
print("문서-단어 행렬", dtm.shape, "| 비어 있는 문서", int((dtm.sum(axis=1) == 0).sum()))'''),
    ("md", """## 3. 토픽 수 선택 — c_v coherence와 보류 집합 perplexity

- 계산: K마다 `LatentDirichletAllocation(n_components=K, random_state=42, learning_method="batch", max_iter=20)`을 학습 집합(90%)에 적합. 토픽별 상위 10단어로 c_v coherence(gensim)를, 보류 10%로 perplexity를 구한다
- 선택 규칙: coherence 최대 K. 두 K가 0.01 안이면 작은 K. perplexity는 참고로 함께 보인다
- 출력: `analysis/output/tables/lda_model_selection.csv`, `analysis/output/figures/fig2a_lda_model_selection.png`
"""),
    ("code", '''from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
K_GRID = [10, 15, 20, 25, 30]
idx_tr, idx_te = train_test_split(np.arange(dtm.shape[0]), test_size=0.1, random_state=42)
gdict = Dictionary(tokens)
def top_words(model, n=10):
    return [[vocab[j] for j in comp.argsort()[::-1][:n]] for comp in model.components_]
sel, models = [], {}
for k in K_GRID:
    m = LatentDirichletAllocation(n_components=k, random_state=42, learning_method="batch", max_iter=20, n_jobs=-1).fit(dtm[idx_tr])
    coh = CoherenceModel(topics=top_words(m), texts=tokens, dictionary=gdict, coherence="c_v", topn=10, processes=1).get_coherence()
    sel.append({"K": k, "c_v": round(coh, 4), "perplexity_heldout": round(m.perplexity(dtm[idx_te]), 1)}); models[k] = m
    print(sel[-1])
sel = pd.DataFrame(sel).set_index("K"); sel.to_csv(TAB / "lda_model_selection.csv")
best = int(sel.c_v.idxmax())
for k in sel.index:
    if sel.c_v[best] - sel.c_v[k] <= 0.01 and k < best:
        best = int(k); break
K = best; print("선택한 K =", K)
fig, a1 = plt.subplots(figsize=(5, 2.8)); a1.plot(sel.index, sel.c_v, "o-", label="c_v coherence"); a1.set_xlabel("K"); a1.set_ylabel("c_v")
a2 = a1.twinx(); a2.plot(sel.index, sel.perplexity_heldout, "s--", color="gray", label="held-out perplexity"); a2.set_ylabel("perplexity")
a1.axvline(K, color="#C0392B", lw=1, ls=":"); a1.set_title("LDA model selection"); h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a1.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(FIG / "fig2a_lda_model_selection.png", dpi=160, bbox_inches="tight"); plt.show()'''),
    ("md", """## 4. 최종 모델 — 전체 문서로 다시 적합, 상위 단어, 문서 배정

- 계산: 선택한 K로 전체 문서에 적합. 토픽별 상위 12단어. 문서마다 확률 최대 토픽을 배정하고 최대 확률도 남긴다
- 라벨: 상위 두 단어(예: `gui·mobile`). 사람이 지은 이름은 쓰지 않는다(D20)
- 출력: `analysis/output/tables/lda_topics_K{K}.csv`, `lda_doc_topics.csv`
"""),
    ("code", '''lda = LatentDirichletAllocation(n_components=K, random_state=42, learning_method="batch", max_iter=25, n_jobs=-1).fit(dtm)
theta = lda.transform(dtm)
allc["lda_topic"] = theta.argmax(axis=1); allc["lda_prob"] = theta.max(axis=1)
tw = top_words(lda, 12)
lda_topics = pd.DataFrame({"topic": range(K), "label": [f"{w[0]}·{w[1]}" for w in tw], "top_words": [", ".join(w) for w in tw],
                           "n_docs": pd.Series(allc.lda_topic).value_counts().reindex(range(K), fill_value=0).values})
lda_topics["share_pct"] = (100 * lda_topics.n_docs / len(allc)).round(2)
lda_topics.to_csv(TAB / f"lda_topics_K{K}.csv", index=False)
allc[["id", "year", "month", "lda_topic", "lda_prob"]].to_csv(TAB / "lda_doc_topics.csv", index=False)
lda_label = lambda t: lda_topics.label[t]
pd.set_option("display.max_colwidth", 160); lda_topics.sort_values("n_docs", ascending=False)'''),
    ("md", """## 5. pyLDAvis — 토픽 간 거리와 단어 분포(대화형)

- 계산: 참고 코드와 같은 `pyLDAvis.lda_model.prepare(lda, dtm, cv)`. HTML로 저장한다(원고에는 넣지 않고 탐색·발표용)
- 출력: `analysis/output/figures/lda_pyldavis_K{K}.html`
"""),
    ("code", '''try:
    import pyLDAvis, pyLDAvis.lda_model
    vis = pyLDAvis.lda_model.prepare(lda, dtm, cv, mds="mmds", sort_topics=False)
    pyLDAvis.save_html(vis, str(FIG / f"lda_pyldavis_K{K}.html")); print("saved", FIG / f"lda_pyldavis_K{K}.html")
except Exception as e:
    print("pyLDAvis 생략:", e)'''),
    ("md", """## 6. 투영 — 한중일어 대상 집합의 토픽 분포 대 세계 전체

- 계산: 토픽별로 세계 전체 비중, 언어 집합의 관측 편수, 기대 편수(= 세계 비중 × 집합 크기), lift(관측/기대, 10건 미만은 NaN). 두 번째 기준선: 한중일 이외 언어명이나 multilingual/cross-lingual을 적은 세계 논문의 토픽 비중 × 집합 크기(exp_lm). 토픽별 언어명 언급률(lang_mention_pct_of_topic)도 남긴다
- 순열 검정: 집합 라벨을 2,000회 무작위 배정해 "차이 없음"일 때의 JSD(집합 대 나머지) 분포를 만들고 관측 JSD와 비교(p)
- 출력: `analysis/output/tables/lda_project_lang.csv`, `analysis/output/figures/fig2b_lda_obs_vs_exp_lang.png`
"""),
    ("code", '''from scipy.spatial.distance import jensenshannon
T = list(range(K)); share_all = allc.lda_topic.value_counts(normalize=True).reindex(T, fill_value=0)
groups = {l: allc[f"lang_{l}"].values for l in LANGS}
# 두 번째 기준선(리뷰 2026-10-14): "한중일 이외의 언어명 또는 multilingual/cross-lingual을 적은 세계 논문"의 토픽 분포.
# 언어명이 적히는 논문 자체가 평가·대화 토픽에 몰리므로, 이 기준선으로 비교하면 한중일 집합만의 결손과 '그 토픽 논문은 언어명을 안 적는다'는 표본틀의 성질을 가른다.
OTHER_LANGS = ["english", "spanish", "french", "german", "italian", "portuguese", "russian", "arabic", "hindi", "bengali", "urdu", "vietnamese", "thai", "indonesian", "malay", "tagalog", "filipino", "turkish", "persian", "farsi", "hebrew", "greek", "dutch", "swedish", "danish", "norwegian", "finnish", "polish", "czech", "hungarian", "romanian", "ukrainian", "swahili", "yoruba", "hausa", "amharic", "tamil", "telugu", "marathi", "gujarati", "punjabi", "kannada", "malayalam", "nepali", "sinhala", "burmese", "khmer", "lao", "mongolian", "tibetan", "uyghur", "kazakh", "uzbek", "basque", "catalan", "galician", "welsh", "irish", "icelandic", "estonian", "latvian", "lithuanian", "slovak", "slovenian", "croatian", "serbian", "bulgarian", "macedonian", "albanian", "georgian", "armenian", "azerbaijani", "afrikaans", "zulu", "xhosa", "igbo", "somali", "latin", "sanskrit", "hokkien", "cantonese", "javanese", "sundanese", "cebuano", "hmong", "quechua", "maori", "hawaiian"]
LM_RE = re.compile(r"\\b(" + "|".join(OTHER_LANGS) + r")\\b|multi-?lingual|cross-?lingual")
_txt = (allc.title.fillna("") + " " + allc.abstract.fillna("")).str.lower()
allc["lang_mention"] = _txt.map(lambda x: bool(LM_RE.search(x)))
allc["cjk_signal"] = allc["cjk_signal"].astype(bool)
lm = allc[allc.lang_mention & ~allc.cjk_signal]; share_lm = lm.lda_topic.value_counts(normalize=True).reindex(T, fill_value=0)   # 한중일 신호 논문은 기준선에서 뺀다(모의 심사: 부분 동어반복)
print(f"언어명(한중일 제외)·multilingual 언급 논문(한중일 신호 없는 것) {len(lm):,}편({100*len(lm)/len(allc):.1f}%) — 두 번째 기준선")
cols = {"label": [lda_label(t) for t in T], "n_all": allc.lda_topic.value_counts().reindex(T, fill_value=0).values, "share_all_pct": (100 * share_all).round(2).values,
        "n_lang_mention": lm.lda_topic.value_counts().reindex(T, fill_value=0).values, "lang_mention_pct_of_topic": (100 * lm.lda_topic.value_counts().reindex(T, fill_value=0) / allc.lda_topic.value_counts().reindex(T)).round(2).values}
for l, m in groups.items():
    obs = allc[m].lda_topic.value_counts().reindex(T, fill_value=0); exp = share_all * m.sum(); exp_lm = share_lm * m.sum()
    cols[f"obs_{l}"] = obs.values; cols[f"exp_{l}"] = exp.round(1).values; cols[f"lift_{l}"] = (obs / exp).round(2).where(obs >= MIN_N).values
    cols[f"exp_lm_{l}"] = exp_lm.round(1).values; cols[f"ratio_lm_{l}"] = (obs / exp_lm.replace(0, np.nan)).round(2).values
t_lang = pd.DataFrame(cols, index=pd.Index(T, name="topic")); t_lang.to_csv(TAB / "lda_project_lang.csv")
from scipy.stats import spearmanr
print("토픽별 언어명 언급률 대 중국어 관측/기대 Spearman:", spearmanr(t_lang.lang_mention_pct_of_topic, t_lang.obs_zh / t_lang.exp_zh.replace(0, np.nan), nan_policy="omit"))
rng = np.random.default_rng(42); top_arr = allc.lda_topic.values
def jsd_vs_rest(mask):
    pa = np.bincount(top_arr[mask], minlength=K) / max(mask.sum(), 1); pb = np.bincount(top_arr[~mask], minlength=K) / max((~mask).sum(), 1)
    return float(jensenshannon(pa, pb, base=2) ** 2)
perm = {}
for l, m in groups.items():
    n = int(m.sum()); point = jsd_vs_rest(m)
    null = np.array([jsd_vs_rest(np.isin(np.arange(len(m)), rng.choice(len(m), n, replace=False))) for _ in range(2000)])
    perm[f"lang:{l}"] = {"n": n, "jsd_vs_rest": round(point, 4), "null_mean": round(float(null.mean()), 4), "null_q95": round(float(np.quantile(null, 0.95)), 4), "p_perm": round(float((np.sum(null >= point) + 1) / 2001), 4)}
print(json.dumps(perm, indent=1)); display(t_lang.sort_values("n_all", ascending=False))
import importlib
sys.path.insert(0, str(ROOT / "analysis/scripts"))
paper_figures = importlib.import_module("19_plot_paper_figures")
paper_figures.plot_language_topics(t_lang, {l: int(groups[l].sum()) for l in LANGS}, FIG / "fig2b_lda_obs_vs_exp_lang.png")'''),
    ("md", """## 7. 연도별 토픽 점유율(세계 전체)

- 계산: 연도 × 토픽 점유율(열 합 1). 2026은 1~9월. 2026 대 2023 차이(퍼센트포인트)로 정렬
- 출력: `analysis/output/tables/lda_year_share.csv`, `analysis/output/figures/fig2c_lda_year_share.png`
"""),
    ("code", '''yt = pd.crosstab(allc.lda_topic, allc.year).reindex(index=T, columns=YEARS, fill_value=0)
yt_share = (100 * yt / yt.sum()).round(2); yt_share["pp_2026_vs_2023"] = (yt_share[2026] - yt_share[2023]).round(2)
yt_share.insert(0, "label", [lda_label(t) for t in T]); yt_share.to_csv(TAB / "lda_year_share.csv")
display(yt_share.sort_values("pp_2026_vs_2023", ascending=False))
M = yt_share.sort_values("pp_2026_vs_2023", ascending=False); fig, ax = plt.subplots(figsize=(6, 0.28 * K + 1.2))
im = ax.imshow(M[YEARS].values, cmap="Blues", aspect="auto"); ax.set_xticks(range(4)); ax.set_xticklabels([str(y) if y != 2026 else "2026\\n(Jan-Sep)" for y in YEARS])
ax.set_yticks(range(K)); ax.set_yticklabels([f"T{t} {M.label[t]}" for t in M.index], fontsize=7)
for i in range(K):
    for j in range(4): ax.text(j, i, f"{M[YEARS].values[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="white" if M[YEARS].values[i, j] > M[YEARS].values.max() * 0.6 else "black")
plt.colorbar(im, ax=ax, fraction=0.03, label="share of year (%)"); ax.set_title("LDA topic share by year (world)"); plt.tight_layout(); plt.savefig(FIG / "fig2c_lda_year_share.png", dpi=160, bbox_inches="tight"); plt.show()'''),
    ("md", """## 7b. 세계 상위·중위·하위 토픽과 언어 집합의 자리 (기간 2023-01 ~ 2026-09 하나로)

- 계산: 토픽을 세계 전체 편수(2023~2026-09 합) 순으로 세워 상위(1~8위)·중위(9~17위)·하위(18~25위)로 나눈다. 칸마다 총 편수와 네 해(2023/2024/2025/2026)의 편수를 그대로 적는다. 세계는 연도별 점유율(열 합 100%)과 그 네 점에 맞춘 직선 기울기(pp/년)를 한 열로 둔다 — 오름(+0.5pp/년 이상), 내림(−0.5 이하), 유지.
- 언어 집합(한국어·중국어·일본어)은 총 관측 편수, 네 해의 편수, 기대 편수(세계 비중 × 집합 크기), 그리고 관측/기대 판정(기대의 1.5배 초과 = 많음, 절반 미만 = 적음)을 둔다. 집합이 작아 연도별 점유율은 쓰지 않는다.
- "비어 있는 자리" = 세계 상위 또는 오르는 토픽인데 그 언어 집합의 관측이 기대의 절반 미만.
- 입력: 7절의 `yt`(연도 × 토픽 편수), 6절의 `t_lang`
- 출력: `analysis/output/tables/lda_world_topic_table.csv`, `lda_world_topic_table.md`(원고용), `analysis/output/figures/fig2d_lda_world_topics.png`
- 주의: 2026은 9개월치이므로 편수가 아니라 점유율로 기울기를 잰다. 2023년은 689편(토픽당 평균 28편)이라 점유율이 거칠다. 토픽 경계는 seed에 따라 흔들리므로(9절) 세부 순위보다 상·중·하위 묶음으로 읽는다.
"""),
    ("code", '''yt_pct = 100 * yt / yt.sum()
wd = pd.DataFrame({"label": [lda_label(t) for t in T], "world_total": yt.loc[T, YEARS].sum(axis=1).values}, index=pd.Index(T, name="topic"))
wd["world_share_pct"] = (100 * wd.world_total / wd.world_total.sum()).round(2)
for y in YEARS:
    wd[f"world_n_{y}"] = yt.loc[T, y].values; wd[f"world_share_{y}"] = yt_pct.loc[T, y].round(1).values
wd["world_slope_pp_per_year"] = [round(float(np.polyfit(YEARS, yt_pct.loc[t, YEARS].values.astype(float), 1)[0]), 2) for t in T]
wd["world_trend"] = np.where(wd.world_slope_pp_per_year >= 0.5, "rising", np.where(wd.world_slope_pp_per_year <= -0.5, "falling", "flat"))
wd = wd.sort_values("world_total", ascending=False); wd["world_rank"] = range(1, K + 1)
wd["world_tier"] = np.where(wd.world_rank <= 8, "top", np.where(wd.world_rank <= 17, "middle", "bottom"))
for l in LANGS:
    m = allc[f"lang_{l}"].values
    for y in YEARS:
        wd[f"{l}_n_{y}"] = [int(((allc.lda_topic == t) & m & (allc.year == y).values).sum()) for t in wd.index]
    wd[f"{l}_obs"] = t_lang.loc[wd.index, f"obs_{l}"].values; wd[f"{l}_exp"] = t_lang.loc[wd.index, f"exp_{l}"].values
    wd[f"{l}_gap"] = np.where(wd[f"{l}_obs"] > wd[f"{l}_exp"] * 1.5, "over", np.where(wd[f"{l}_obs"] < wd[f"{l}_exp"] * 0.5, "under", "near"))
    wd[f"{l}_exp_lm"] = t_lang.loc[wd.index, f"exp_lm_{l}"].values
    wd[f"{l}_gap_lm"] = np.where(wd[f"{l}_exp_lm"] < 1, "n/a", np.where(wd[f"{l}_obs"] > wd[f"{l}_exp_lm"] * 1.5, "over", np.where(wd[f"{l}_obs"] < wd[f"{l}_exp_lm"] * 0.5, "under", "near")))
wd["lang_mention_pct"] = t_lang.loc[wd.index, "lang_mention_pct_of_topic"].values
wd["empty_seat"] = [", ".join(LANG_NAME[l] for l in LANGS if (r.world_tier == "top" or r.world_trend == "rising") and getattr(r, f"{l}_gap") == "under") for r in wd.itertuples()]
wd.to_csv(TAB / "lda_world_topic_table.csv")
print("세계 상위 8:", ", ".join(f"{r.label}({int(r.world_total):,}, {r.world_slope_pp_per_year:+.2f}pp/년)" for r in wd[wd.world_tier == "top"].itertuples()))
print("세계 하위 8:", ", ".join(f"{r.label}({int(r.world_total):,}, {r.world_slope_pp_per_year:+.2f}pp/년)" for r in wd[wd.world_tier == "bottom"].itertuples()))
print("비어 있는 자리(상위 또는 오르는 토픽에서 기대의 절반 미만):"); print(wd[wd.empty_seat != ""][["label", "world_tier", "world_trend", "empty_seat"]].to_string())

# 원고용 Markdown 표
def ycounts(r, s): return "/".join(str(int(r[f"{s}_n_{y}"])) for y in YEARS)
def wcell(r): return f"**{int(r['world_total']):,}** ({r['world_share_pct']:.1f}%) · {ycounts(r, 'world')} · 점유율 " + "/".join(f"{r[f'world_share_{y}']:.1f}" for y in YEARS) + f"% · 기울기 {r['world_slope_pp_per_year']:+.2f}pp/년"
def lcell(r, l): return f"**{int(r[f'{l}_obs'])}** · {ycounts(r, l)} · 기대 {r[f'{l}_exp']:.1f}" + (" · 많음" if r[f"{l}_gap"] == "over" else " · 적음" if r[f"{l}_gap"] == "under" else "")
sizes = {"world": [int((allc.year == y).sum()) for y in YEARS], **{l: [int((allc[f"lang_{l}"] & (allc.year == y)).sum()) for y in YEARS] for l in LANGS}}
head = ("| 순위 | 토픽 | 세계 (총 " + f"{len(allc):,}; " + "/".join(f"{n:,}" for n in sizes["world"]) + ") | " + " | ".join(f"{LANG_NAME[l]} (총 {sum(sizes[l])}; " + "/".join(str(n) for n in sizes[l]) + ")" for l in LANGS) + " |\\n|---|---|---|---|---|---|\\n")
def block(df): return head + "".join(f"| {int(r['world_rank'])} | {r['label']} | {wcell(r)} | " + " | ".join(lcell(r, l) for l in LANGS) + " |\\n" for _, r in df.iterrows())
md = "칸 읽는 법: **총 편수(2023-01~2026-09)** (세계 비중 %) · 연도별 편수 2023/2024/2025/2026(1~9월) · (세계만) 연도별 점유율 %와 네 점 직선 기울기 pp/년 · (언어 집합) 기대 = 세계 비중 × 집합 크기, 많음 = 기대의 1.5배 초과, 적음 = 기대의 절반 미만. 네 집합 모두 같은 자료·같은 기간.\\n\\n"
for tier, name in (("top", "세계 상위 토픽(1~8위)"), ("middle", "세계 중위 토픽(9~17위)"), ("bottom", "세계 하위 토픽(18~25위)")):
    md += f"### {name}\\n\\n" + block(wd[wd.world_tier == tier]) + "\\n"
(TAB / "lda_world_topic_table.md").write_text(md); print("saved", TAB / "lda_world_topic_table.md")

fig, ax = plt.subplots(figsize=(8, 0.3 * K + 1.2)); y = np.arange(K)
ax.barh(y, wd.world_share_pct, color=["#C0392B" if d == "rising" else "#2E86C1" if d == "falling" else "#B0B7C3" for d in wd.world_trend])
ax.set_yticks(y); ax.set_yticklabels([f"{int(r.world_rank)}. {r.label}" for r in wd.itertuples()], fontsize=7); ax.invert_yaxis(); ax.set_xlabel("world share of LLM-agent papers, 2023-01 to 2026-09 (%)")
for i, r in enumerate(wd.itertuples()):
    ax.text(r.world_share_pct + 0.1, i, f"{r.world_slope_pp_per_year:+.1f}pp/yr   " + " ".join(f"{LANG_NAME[l][:2]}:{getattr(r, f'{l}_gap')[0].upper()}" for l in LANGS), va="center", fontsize=6.5, color="gray")
for cut in (7.5, 16.5): ax.axhline(cut, color="k", lw=0.6, ls=":")
ax.set_title("World topics by size (bar colour = trend: red rising, blue falling); right: slope and Ko/Ch/Ja gap (O over, U under, N near)", fontsize=8.5)
plt.tight_layout(); plt.savefig(FIG / "fig2d_lda_world_topics.png", dpi=160, bbox_inches="tight"); plt.show()'''),
    ("md", """## 8. 언어 집합의 상위 단어(문서 빈도)와 세계 대비 비율

- 계산: 언어 집합 안에서 표제어의 문서 빈도(%)와 세계 전체 문서 빈도(%)의 비율. 집합에서 5편 이상 나타난 단어만. 토픽 이름 대신 자료에서 나온 단어로 집합의 성격을 적는다(D20)
- 출력: `analysis/output/tables/lda_lang_subset_keywords.csv`
"""),
    ("code", '''docs_sets = [set(t) for t in tokens]; n_all = len(docs_sets)
df_all = pd.Series([w for d in docs_sets for w in d]).value_counts()
rows = []
for l, m in groups.items():
    sub = [docs_sets[i] for i in np.where(m)[0]]; n = len(sub)
    df_sub = pd.Series([w for d in sub for w in d]).value_counts()
    df_sub = df_sub[df_sub >= 5]
    r = pd.DataFrame({"lang": LANG_NAME[l], "word": df_sub.index, "doc_pct_lang": (100 * df_sub / n).round(1).values, "doc_pct_world": (100 * df_all.reindex(df_sub.index).fillna(0) / n_all).round(2).values})
    r["ratio"] = (r.doc_pct_lang / r.doc_pct_world.replace(0, np.nan)).round(1); rows.append(r.sort_values("ratio", ascending=False).head(20))
subset_kw = pd.concat(rows); subset_kw.to_csv(TAB / "lda_lang_subset_keywords.csv", index=False)
for l in LANGS: print(LANG_NAME[l], "→", ", ".join(f"{w}({d:.0f}%)" for w, d in subset_kw[subset_kw.lang == LANG_NAME[l]][["word", "doc_pct_lang"]].head(15).values))'''),
    ("md", """## 9. 안정성 — seed 3개

- 계산: random_state 43/44로 같은 K를 다시 적합하고 토픽별 상위 10단어 집합의 최대 Jaccard. 평균 0.5 미만 토픽은 "불안정"으로 표시
"""),
    ("code", '''base_sets = [set(w) for w in top_words(lda, 10)]; stab = []
for seed in [43, 44]:
    m = LatentDirichletAllocation(n_components=K, random_state=seed, learning_method="batch", max_iter=25, n_jobs=-1).fit(dtm)
    other = [set(w) for w in top_words(m, 10)]; stab.append([max(len(a & b) / len(a | b) for b in other) for a in base_sets])
stab = pd.DataFrame(stab, index=["seed43", "seed44"], columns=[f"T{t}" for t in T]).T
stab["mean_jaccard"] = stab.mean(axis=1).round(2); stab["stable"] = stab.mean_jaccard >= 0.5
print("안정 토픽", int(stab.stable.sum()), "/", K); stab'''),
    ("md", """## 9b. 시드별 연도 점유율 기울기 — 오름·내림 방향이 시드를 바꿔도 반복되는가(리뷰 2026-10-14)

- 계산: 시드 43·44 모델의 문서 배정으로 연도 점유율 기울기(pp/년)를 다시 구하고, 각 기준 토픽(seed 42)에 상위 10단어 자카드가 가장 큰 시드 토픽을 대응시켜 기울기 부호를 비교한다. 세 모델 모두 +0.5pp/년 이상 또는 모두 −0.5pp/년 이하이면 "반복"으로 표시한다.
- 2026년 1위 토픽(benchmark·score)의 월별 점유율도 남긴다(1월→9월).
- 출력: `analysis/output/tables/lda_slope_by_seed.csv`
"""),
    ("code", '''def year_slopes(model):
    assign = model.transform(dtm).argmax(axis=1); ct = pd.crosstab(assign, allc.year.values).reindex(columns=YEARS, fill_value=0); pct = 100 * ct / ct.sum()
    return {t: round(float(np.polyfit(YEARS, pct.loc[t, YEARS].values.astype(float), 1)[0]), 2) if t in pct.index else np.nan for t in range(K)}, model
base_slope = {t: float(wd.loc[t, "world_slope_pp_per_year"]) for t in T}; rows = []
seed_models = {}
for seed in [43, 44]:
    m = LatentDirichletAllocation(n_components=K, random_state=seed, learning_method="batch", max_iter=25, n_jobs=-1).fit(dtm); seed_models[seed] = m
    sl, _ = year_slopes(m); other = [set(w) for w in top_words(m, 10)]
    for t in T:
        j = int(np.argmax([len(base_sets[t] & o) / len(base_sets[t] | o) for o in other])); rows.append({"topic": t, "seed": seed, "matched": j, "jaccard": round(len(base_sets[t] & other[j]) / len(base_sets[t] | other[j]), 2), "slope": sl[j]})
SS = pd.DataFrame(rows).pivot(index="topic", columns="seed", values=["slope", "jaccard"]); SS.columns = [f"{a}_{b}" for a, b in SS.columns]
SS["slope_42"] = [base_slope[t] for t in SS.index]; SS["label"] = [lda_label(t) for t in SS.index]
sign = lambda x: "+" if x >= 0.5 else "-" if x <= -0.5 else "0"
SS["direction_42_43_44"] = [sign(r.slope_42) + sign(r.slope_43) + sign(r.slope_44) for r in SS.itertuples()]
SS["repeated"] = SS.direction_42_43_44.isin(["+++", "---"]); SS = SS.sort_values("slope_42", ascending=False); SS.to_csv(TAB / "lda_slope_by_seed.csv")
print("세 시드 모두 같은 방향(|기울기| ≥ 0.5pp/년):"); print(SS[SS.repeated][["label", "slope_42", "slope_43", "slope_44", "jaccard_43", "jaccard_44"]].to_string())
top_t = int(wd.index[0]); m26 = allc[allc.year == 2026]; monthly_top = (100 * m26.groupby("month").lda_topic.apply(lambda s: (s == top_t).mean())).round(1)
print(f"\\n1위 토픽 {lda_label(top_t)} 2026년 월별 점유율(%):", monthly_top.to_dict())
slope_by_seed = {str(t): {"label": r.label, "slope_42": r.slope_42, "slope_43": r.slope_43, "slope_44": r.slope_44, "jaccard_43": r.jaccard_43, "jaccard_44": r.jaccard_44, "direction": r.direction_42_43_44, "repeated": bool(r.repeated)} for t, r in SS.iterrows()}'''),
    ("md", """## 10. 정본 저장

- 저장: K 선택표, 토픽 단어·라벨·건수, 투영 표(관측·기대·lift), 순열 검정, 연도 점유율, 집합 상위 단어, 안정성 → `verification/topics_lda_YYYYMMDD.json`
"""),
    ("code", '''out = {"generated": TODAY, "notebook": "analysis/notebooks/02_lda.ipynb", "period": counts["definitions"]["period"], "year_basis": YEAR_BASIS,
       "n_docs": int(len(allc)), "vocab_size": int(len(vocab)), "vectorizer": {"min_df": 10, "max_df": 0.5, "tokens": "spaCy en_core_web_sm lemma, alpha, len>2, stopwords removed"},
       "domain_stopwords": sorted(DOMAIN_STOP), "model_selection": {str(k): {c: float(v) for c, v in r.items()} for k, r in sel.iterrows()}, "K": int(K),
       "lda_params": {"random_state": 42, "learning_method": "batch", "max_iter": 25},
       "topics": {int(r.topic): {"label": r.label, "top_words": r.top_words.split(", "), "n_docs": int(r.n_docs), "share_pct": float(r.share_pct)} for r in lda_topics.itertuples()},
       "project_lang": {str(t): {c: (None if pd.isna(v) else (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v))) for c, v in r.items()} for t, r in t_lang.iterrows()},
       "permutation": perm,
       "year_share_pct": {str(t): {str(y): float(yt_share.loc[t, y]) for y in YEARS} for t in T},
       "world_topic_table": {str(t): {c: (None if pd.isna(v) else (v if isinstance(v, str) else float(v))) for c, v in r.items()} for t, r in wd.iterrows()},
       "lang_subset_keywords": {LANG_NAME[l]: subset_kw[subset_kw.lang == LANG_NAME[l]][["word", "doc_pct_lang", "doc_pct_world", "ratio"]].to_dict(orient="records") for l in LANGS},
       "stability_mean_jaccard": {k: float(v) for k, v in stab.mean_jaccard.items()},
       "slope_by_seed": slope_by_seed, "top_topic_2026_monthly_share_pct": {str(k): float(v) for k, v in monthly_top.items()},
       "lang_mention_baseline": {"definition": "title+abstract contains a non-CJK language name or multilingual/cross-lingual, and no CJK language-name signal", "n": int(len(lm)), "share_pct": round(100 * len(lm) / len(allc), 2)}}
p = VER / f"topics_lda_{TODAY}.json"; p.write_text(json.dumps(out, ensure_ascii=False, indent=1)); print("saved", p)'''),
])

# ─────────────────────────────────────────────────────────────────────────────
# 03 BERTopic
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 04 agent links (탐색, D17)
# ─────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────
# 05 HCLT 논문집 분석(D23·D24·D26): 연도별 건수, 사람 판정 대 형태소 판정, 같은 사전 키워드군 비교, 상위 명사
# ──────────────────────────────────────────────────────────────────────────
nb05 = nb([
    ("md", """# 05. HCLT(한글 및 한국어 정보처리 학술대회) 논문집의 LLM 에이전트 연구

- 질문: 국내 학회(HCLT 2023~2026)에서 LLM 에이전트 논문은 연도별로 몇 편이고, arXiv(세계 전체·대상 언어)와 키워드군 분포가 어떻게 다른가 (4.6절 후보).
- 입력
  - `preprocessing/output/hclt_papers.csv` — 07: 목차 명부 525편의 제목·저자·쪽수와 선정 판정. 요약·OCR 텍스트는 제외한다.
  - `analysis/output/hclt_tokens.csv` — 논문별 키워드군과 형태소 기반 에이전트 판정. 형태소 텍스트는 제외한다.
  - `analysis/output/arxiv_lemma_families_2023_2026q3.csv` — 09: arXiv LLM 에이전트 논문 전체(2023-01~2026-09)에 같은 사전을 spaCy 표제어로 적용한 키워드군.
  - `verification/review_review_hclt_agent_*.jsonl` — 사람 판정(D23, 검수자 song, 62편 전수).
  - 정본: `verification/hclt_counts_*.json`(07), `verification/hclt_morph_*.json`(09).
- 주의
  - 판정의 정본은 **사람 판정**이다. 형태소 판정은 검수 입력을 고르는 선별기이며 정밀도·재현율만 보고한다.
  - 본 비교 기간은 arXiv와 같은 2023-01~2026-09(D30). HCLT 2026 논문집도 9월까지의 원고다.
  - 키워드군은 "같은 사전의 영문(표제어)·국문(형태소) 판"으로 매칭했다(D26). 영문만 있는 arXiv와 국문 위주인 HCLT의 차이는 사전 번역의 차이를 포함한다.
"""),
    ("code", '''import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "verification").exists())   # 노트북은 analysis/notebooks/ 에 있다
DATA = ROOT / "preprocessing/output"; VER = ROOT / "verification"
FIG = ROOT / "analysis/output/figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "analysis/output/tables"; TAB.mkdir(parents=True, exist_ok=True)
TODAY = pd.Timestamp.today().strftime("%Y%m%d")
try:
    import sys; sys.path.insert(0, str(ROOT / "analysis/scripts")); import figure_style; figure_style.apply_style(); C = figure_style.COLORS; CYC = figure_style.COLOR_CYCLE
except Exception:
    C = {"green": "#46B07B", "orange": "#FF9E02", "teal": "#3A9E8F", "coral": "#E87B4A", "green_dark": "#2D7A53", "orange_dark": "#C47800", "green_light": "#8DD4AA", "orange_light": "#FFD080", "green_pale": "#C8EBDA", "orange_pale": "#FFECC0", "gray": "#7F8C8D", "gray_light": "#BDC3C7", "gray_dark": "#4A4A4A", "text": "#333333"}; CYC = [C["green"], C["orange"], C["teal"], C["coral"]]
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
from matplotlib import font_manager as _fm
_ko = [f for f in ("AppleGothic", "NanumGothic", "Malgun Gothic") if f in {x.name for x in _fm.fontManager.ttflist}]
plt.rcParams["font.family"] = ["DejaVu Sans"] + _ko[:1]; plt.rcParams["axes.unicode_minus"] = False   # 그림 글자는 영문이므로 기본 글꼴을 앞에 둔다(한글은 뒤로)
YEARS = [2023, 2024, 2025, 2026]   # D30: 본 비교 기간 2023~2026(HCLT 2026 논문집 포함)

h = pd.read_csv(DATA / "hclt_papers.csv"); OUTA = ROOT / "analysis/output"; SUFFIX = "2023_2026q3"
tok = pd.read_csv(OUTA / "hclt_tokens.csv"); ax = pd.read_csv(OUTA / f"arxiv_lemma_families_{SUFFIX}.csv", dtype={"id": str})
counts = json.loads(Path(sorted(glob.glob(str(VER / "hclt_counts_*.json")))[-1]).read_text())
morph = json.loads(Path(sorted(glob.glob(str(VER / "hclt_morph_*.json")))[-1]).read_text())
rev = pd.read_json(sorted(glob.glob(str(VER / "review_review_hclt_agent_*.jsonl")))[-1], lines=True)
h["human"] = h.id.isin(set(rev[rev.final == "LLM 에이전트"].id)); h = h.merge(tok[["id", "is_llm_agent_morph"] + [c for c in tok.columns if c.startswith("f_")]], on="id")
FAMS = [c[2:] for c in ax.columns if c.startswith("f_")]

# 정본 대조
assert h.groupby("year").size().to_dict() == {int(k): v["papers"] for k, v in counts["human_review"]["by_year"].items()}, "연도별 논문 수가 정본과 다르다"
assert h[h.human].groupby("year").size().to_dict() == {int(k): v for k, v in morph["hclt"]["llm_agent_human_by_year"].items()}, "사람 판정 건수가 정본과 다르다"
assert h[h.is_llm_agent_morph].groupby("year").size().to_dict() == {int(k): v for k, v in morph["hclt"]["llm_agent_morph_by_year"].items()}
assert len(ax) == morph["n"]["arxiv_world"]
assert h.human.sum() == len(rev) == 62 and rev.held.sum() == 0
print("정본 대조 통과: HCLT", len(h), "편 | 사람 판정 LLM 에이전트", int(h.human.sum()), "| arXiv", len(ax))'''),
    ("md", """## 1. 연도별 논문 수와 LLM 에이전트 논문 수

- 계산: 목차 명부의 논문 수, 사람 판정(D23)으로 확정한 LLM 에이전트 수, 비율. 비교용으로 arXiv 세계 전체의 LLM 에이전트 수(정본 `corpus_counts`)를 같은 표에 둔다.
- 입력: `h`(사람 판정 열 human), `verification/corpus_counts_*.json`.
- 주의: HCLT는 자연어처리 학회이고 arXiv는 다섯 분류(cs.AI/CL/CV/CY/LG) 전체라 분모의 성격이 다르다. 비율은 "그 자료 안에서의 비중"이다.
"""),
    ("code", '''cc = json.loads(Path(sorted(glob.glob(str(VER / "corpus_counts_*.json")))[-1]).read_text())
t1 = h.groupby("year").agg(papers=("id", "size"), llm_agent=("human", "sum")); t1["share_pct"] = (100 * t1.llm_agent / t1.papers).round(1)
t1["arxiv_llm_agent"] = pd.Series({int(k): v for k, v in cc["llm_agent_by_year"].items()})
t1.loc["합계(2023~2026)"] = t1.loc[YEARS].sum(numeric_only=True); t1.loc["합계(2023~2026)", "share_pct"] = round(100 * t1.loc[YEARS].llm_agent.sum() / t1.loc[YEARS].papers.sum(), 1)
t1.to_csv(TAB / "tab_hclt_year_counts.csv"); print(t1.to_string())

plt.rcParams["font.family"] = ["Nanum Gothic"]
plt.rcParams["axes.unicode_minus"] = False
fig, a1 = plt.subplots(figsize=(6.5, 3.4)); yrs = [2023, 2024, 2025, 2026]; x = np.arange(len(yrs))
a1.bar(x - 0.2, t1.loc[yrs].papers, 0.4, label="전체 논문", color=C["gray_light"], zorder=3); a1.bar(x + 0.2, t1.loc[yrs].llm_agent, 0.4, label="LLM 에이전트 논문(저자 판정)", color=C["green"], zorder=3)
for i, y in enumerate(yrs):
    a1.text(x[i] - 0.2, t1.loc[y].papers + 2, f"{t1.loc[y].papers:.0f}", ha="center", fontsize=7.5, color=C["gray_dark"])
    a1.text(x[i] + 0.2, t1.loc[y].llm_agent + 2, f"{t1.loc[y].llm_agent:.0f} ({t1.loc[y].share_pct}%)", ha="center", fontsize=8, color=C["green_dark"], fontweight="bold")
a1.set_xticks(x); a1.set_xticklabels([str(y) for y in yrs]); a1.set_ylabel("논문 수"); a1.set_title("한글 및 한국어 정보처리 학술대회 논문집\\n전체 논문과 LLM 에이전트 논문"); a1.legend(loc="upper left"); a1.set_ylim(0, t1.loc[yrs].papers.max() * 1.15); a1.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(FIG / "fig6_hclt_year_counts.png", dpi=200); plt.show()'''),
    ("md", """## 2. 형태소 기반 판정 대 사람 판정

- 계산: 09의 형태소 판정(agent 군 AND llm 군, Kiwi·spaCy 표제어 대조)을 사람 판정과 비교해 정밀도·재현율을 본다. 놓친 논문의 제목을 나열하고, 놓친 이유(agent 군은 있는데 llm 군이 없는가)를 센다.
- 입력: `tok.is_llm_agent_morph`, `h.human`.
- 주의: 검수 입력이 "agent 단어 포함 62편"이므로 그 밖의 논문에서 사람이 LLM 에이전트로 볼 것이 더 있는지는 이 자료로 알 수 없다(한계).
"""),
    ("code", '''mv = morph["hclt"]["morph_vs_human"]
tp = int((h.human & h.is_llm_agent_morph).sum()); fp = int((~h.human & h.is_llm_agent_morph).sum()); fn = int((h.human & ~h.is_llm_agent_morph).sum())
assert (tp, fp, fn) == (mv["tp"], mv["fp"], mv["fn"]), "정밀도·재현율 근거 수치가 정본과 다르다"
print(f"정밀도 {tp/(tp+fp):.3f}  재현율 {tp/(tp+fn):.3f}  (TP {tp}, FP {fp}, FN {fn})")
miss = h[h.human & ~h.is_llm_agent_morph][["id", "year", "title_ko", "f_agent", "f_llm"]]; print("형태소 판정이 놓친 논문(사람은 LLM 에이전트):"); print(miss.to_string(index=False))
print(); print("놓친 이유(agent 군, llm 군):"); print(miss[["f_agent", "f_llm"]].value_counts())'''),
    ("md", """## 3. 같은 사전으로 본 키워드군 비율: HCLT 대 arXiv(세계 전체·대상 언어)

- 계산: 17개 키워드군이 LLM 에이전트 논문에 나타나는 비율(%). HCLT는 사람 판정 논문 62편(2023~2026 전체, 2023~2025와 2026을 따로도), arXiv는 세계 전체(2023-01~2026-09, id 접두 기준). 09 정본과 대조한다. 소속 국가 열은 쓰지 않는다(D30).
- 입력: `h`(f_* 열), `ax`(f_* 열).
- 주의: HCLT 2023~2025는 24편이라 한 편이 4.2%p다. 건수 열을 함께 읽는다. agent·llm 군은 정의상 거의 100%라 그림에서 뺀다.
"""),
    ("code", '''hall = h[h.human]; hs = h[h.year.isin([2023, 2024, 2025]) & h.human]; h26 = h[(h.year == 2026) & h.human]
rate = lambda d, f: round(100 * d[f"f_{f}"].mean(), 1)
N = {"arXiv world": len(ax), "HCLT 2023-26": len(hall), "HCLT 2023-25": len(hs), "HCLT 2026": len(h26)}; print("n =", N)
t3 = pd.DataFrame({f: {"arXiv world": rate(ax, f), "HCLT 2023-26": rate(hall, f), "HCLT 2023-25": rate(hs, f), "HCLT 2026": rate(h26, f),
                       "n HCLT 23-25": int(hs[f"f_{f}"].sum()), "n HCLT 26": int(h26[f"f_{f}"].sum())} for f in FAMS}).T
for f in FAMS:
    ref = morph["family_rate_pct"][f]
    assert (t3.loc[f, "arXiv world"], t3.loc[f, "HCLT 2023-25"], t3.loc[f, "HCLT 2026"]) == (ref["arxiv_world"], ref["hclt_2023_2025"], ref["hclt_2026"]), f"{f} 비율이 정본과 다르다"
t3 = t3.sort_values("arXiv world", ascending=False); t3.to_csv(TAB / "tab_hclt_vs_arxiv_families.csv"); print(t3.to_string())

show = [f for f in t3.index if f not in ("agent", "llm")]; cols = ["arXiv world", "HCLT 2023-26", "HCLT 2023-25", "HCLT 2026"]; M = t3.loc[show, cols].astype(float)
fig, a = plt.subplots(figsize=(7, 6.2)); im = a.imshow(M.values, cmap="Blues", aspect="auto", vmin=0, vmax=max(40, M.values.max()))
a.set_xticks(range(4)); a.set_xticklabels([f"{c}\\n(n={N[c]:,})" for c in cols], fontsize=9); a.set_yticks(range(len(show))); a.set_yticklabels(show)
for i in range(len(show)):
    for j in range(4):
        a.text(j, i, f"{M.values[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if M.values[i, j] > 30 else "black")
a.set_title("Keyword-family share (%) among LLM-agent papers\\nsame dictionary, lemma/morpheme match"); plt.colorbar(im, ax=a, fraction=0.03, label="%")
plt.tight_layout(); plt.savefig(FIG / "fig7_hclt_vs_arxiv_families.png", dpi=160, bbox_inches="tight"); plt.show()'''),
    ("md", """## 3b. 키워드군 비율: 세계 전체 vs 대상 언어 집합(사람 라벨, D36) vs HCLT

- 계산: 같은 사전·같은 표제어 판정(09)으로 arXiv 세계 전체와 한국어·중국어·일본어 집합(allc의 lang_* = 라벨 '실제 연구 대상')의 키워드군 비율(%)과 편수. HCLT 62편을 나란히 둔다. 원고 표 4(4.4절)의 정본.
- 입력: `ax`(id·f_* 열) + `allc`(lang_* 열), `hall`.
- 주의: 한국어 25편·일본어 18편은 한 편이 4~6%p다. 편수 열을 함께 읽는다. agent·llm 군은 정의상 거의 100%.
"""),
    ("code", '''LANGS = ["ko", "zh", "ja"]; LANG_NAME = {"ko": "Korean", "zh": "Chinese", "ja": "Japanese"}
allc = pd.read_csv(DATA / f"corpus_llm_agent_all_{SUFFIX}.csv", dtype={"id": str}, usecols=["id", "lang_ko", "lang_zh", "lang_ja"])
cc = json.loads(Path(sorted(glob.glob(str(VER / "corpus_counts_*.json")))[-1]).read_text())
assert {l: int(allc[f"lang_{l}"].sum()) for l in LANGS} == cc["cjk_lang_by_lang_total"], "대상 언어 집합 크기가 정본과 다르다"
axl = ax.merge(allc[["id"] + [f"lang_{l}" for l in LANGS]], on="id", how="left")
assert len(axl) == len(allc) and axl[[f"lang_{l}" for l in LANGS]].notna().all().all()
sets = {"arXiv world": axl, **{LANG_NAME[l]: axl[axl[f"lang_{l}"]] for l in LANGS}, "HCLT 2023-26": hall}
NS = {k: len(v) for k, v in sets.items()}; print("n =", NS)
t3b = pd.DataFrame({f: {**{k: rate(v, f) for k, v in sets.items()}, **{f"n {k}": int(v[f"f_{f}"].sum()) for k, v in sets.items() if k != "arXiv world"}} for f in FAMS}).T
t3b = t3b.sort_values("arXiv world", ascending=False); t3b.to_csv(TAB / "tab_families_by_set.csv"); print(t3b.to_string())
family_by_set = {"n": NS, "rows": {f: {c: (float(v) if not c.startswith("n ") else int(v)) for c, v in r.items()} for f, r in t3b.iterrows()}}'''),
    ("md", """## 3b-2. 한 장으로 보는 arXiv 대 HCLT — 키워드군 비율 덤벨 그림

- 계산: 표 3b의 세 집합(arXiv 세계 20,107편, arXiv 한국어 집합 25편, HCLT 62편) 비율을 군마다 한 줄에 세 점으로 찍고 HCLT−세계 차이(pp) 순으로 정렬한다. 점 옆에 편수를 적는다. agent·llm 군 제외.
- 읽는 법: 오른쪽(HCLT가 높음)은 질의응답·번역·평가, 왼쪽(HCLT가 낮음)은 강화학습·물리 환경·안전·도구 호출. 한국어 집합(25편)이 HCLT와 같은 쪽에 있는 군이 "국내 연구의 방향".
- 출력: `analysis/output/figures/fig7c_arxiv_vs_hclt_dumbbell.png`
"""),
    ("code", '''D = t3b.drop(index=["agent", "llm"]).copy(); D["diff"] = D["HCLT 2023-26"] - D["arXiv world"]; D = D.sort_values("diff")
plt.rcParams["font.family"] = ["Nanum Gothic"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(8, 6.2)); y = np.arange(len(D))
for i, (f, r) in enumerate(D.iterrows()):
    lo, hi = min(r["arXiv world"], r["HCLT 2023-26"]), max(r["arXiv world"], r["HCLT 2023-26"])
    ax.plot([lo, hi], [i, i], color=C["green_pale"] if r["diff"] >= 0 else C["orange_pale"], lw=5, zorder=1, solid_capstyle="round")
ax.scatter(D["arXiv world"], y, color=C["gray"], s=55, label=f"arXiv 세계 전체 (n={NS['arXiv world']:,})", zorder=3)
ax.scatter(D["Korean"], y, color=C["orange"], s=55, marker="s", label=f"arXiv 한국어 집합 (n={NS['Korean']})", zorder=3)
ax.scatter(D["HCLT 2023-26"], y, color=C["green_dark"], s=75, marker="D", label=f"한글 및 한국어 정보처리 학술대회 논문집\\n2023–2026년, 저자 판정 (n={NS['HCLT 2023-26']})", zorder=4)
for i, (f, r) in enumerate(D.iterrows()):
    ax.text(r["HCLT 2023-26"] + (1.3 if r["diff"] >= 0 else -1.3), i, f"{int(r['n HCLT 2023-26'])}", va="center", ha="left" if r["diff"] >= 0 else "right", fontsize=7.5, color=C["green_dark"])
ax.set_yticks(y); ax.set_yticklabels([f.replace("_", " ") for f in D.index]); ax.set_xlabel("키워드군이 나타난 LLM 에이전트 논문의 비율 (%)"); ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8); ax.set_title("자료별 키워드군 비율: 논문집과 세계 전체의 차이 순")
plt.tight_layout(); plt.savefig(FIG / "fig7c_arxiv_vs_hclt_dumbbell.png", dpi=200, bbox_inches="tight"); plt.show()
print(D[["arXiv world", "Korean", "HCLT 2023-26", "diff", "n HCLT 2023-26"]].to_string())'''),
    ("md", """## 3c. 키워드군 동시출현 lift — 세계 전체 LLM 에이전트 논문 안에서 어떤 군이 함께 나타나는가

- 계산: 군 X와 Y가 같은 논문에 함께 나타나는 비율을 독립일 때의 기대(P(X)·P(Y))로 나눈 lift = P(X∧Y) / (P(X)·P(Y)). 1보다 크면 함께 나타나는 경향, 작으면 서로 피하는 경향. agent·llm 군은 정의상 거의 모든 논문에 있어 뺀다.
- 입력: `axl`(20,107편, f_* 열). 표 3(원고)의 키워드군과 같은 사전·같은 판정.
- 출력: `analysis/output/tables/tab_family_lift.csv`, `analysis/output/figures/fig7b_family_lift.png`
- 주의: lift는 두 군의 동시출현만 보며 토픽이나 인과를 뜻하지 않는다. 비율이 작은 군(transformer 1.5%, diffusion 1.2%)은 lift가 크게 흔들린다.
"""),
    ("code", '''fam_show = [f for f in t3b.index if f not in ("agent", "llm")]
M = axl[[f"f_{f}" for f in fam_show]].astype(bool).values; pX = M.mean(axis=0)
Mi = M.astype(int); joint = (Mi.T @ Mi) / len(M)   # 불리언 행렬곱은 논리합이 되므로 정수로 센다
lift = joint / np.outer(pX, pX); np.fill_diagonal(lift, np.nan)
LF = pd.DataFrame(lift, index=fam_show, columns=fam_show).round(2); LF.to_csv(TAB / "tab_family_lift.csv")
pairs = [(fam_show[i], fam_show[j], round(float(lift[i, j]), 2), int((M[:, i] & M[:, j]).sum())) for i in range(len(fam_show)) for j in range(i + 1, len(fam_show))]
pairs = sorted(pairs, key=lambda x: -x[2]); print("함께 나타나는 쌍(상위 10):", pairs[:10]); print("서로 피하는 쌍(하위 10):", pairs[-10:])
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
cmap_brand = LinearSegmentedColormap.from_list("brand_div", [C["orange_dark"], C["orange"], "#FFFFFF", C["green"], C["green_dark"]])
fig, ax = plt.subplots(figsize=(9, 8)); im = ax.imshow(np.clip(lift, 0.25, 4), cmap=cmap_brand, norm=TwoSlopeNorm(vmin=0.25, vcenter=1.0, vmax=4)); ax.grid(False)
ax.set_xticks(range(len(fam_show))); ax.set_xticklabels([f.replace("_", " ") for f in fam_show], rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(fam_show))); ax.set_yticklabels([f"{f.replace('_', ' ')} ({100*p:.0f}%)" for f, p in zip(fam_show, pX)], fontsize=8)
for i in range(len(fam_show)):
    for j in range(len(fam_show)):
        if i != j: ax.text(j, i, f"{lift[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="white" if (lift[i, j] > 2.2 or lift[i, j] < 0.5) else C["text"])
cb = plt.colorbar(im, ax=ax, fraction=0.03, ticks=[0.25, 0.5, 1, 2, 4]); cb.set_label("lift (log-like scale; green = co-occur, orange = avoid)"); cb.ax.set_yticklabels(["≤0.25", "0.5", "1", "2", "≥4"])
ax.set_title(f"Keyword-family co-occurrence lift within LLM-agent papers (n={len(M):,}, 2023-01 to 2026-09)")
plt.tight_layout(); plt.savefig(FIG / "fig7b_family_lift.png", dpi=200, bbox_inches="tight"); plt.show()
family_lift = {"families": fam_show, "share_pct": {f: round(100 * float(p), 1) for f, p in zip(fam_show, pX)}, "lift": {a: {b: (None if np.isnan(LF.loc[a, b]) else float(LF.loc[a, b])) for b in fam_show} for a in fam_show}, "top_pairs": pairs[:15], "bottom_pairs": pairs[-15:]}'''),
    ("md", """## 4. 집계 저장

- 계산: 논문별 판정으로 구한 편수·키워드군 비율을 저장한다.
- 입력: 공개한 판정 열과 arXiv 키워드군 자료.
- 주의: 학술대회 요약·형태소 텍스트는 배포하지 않으며, 추출 오류를 교정한 재분석 결과는 아니다.
"""),
    ("code", '''json.dump({"generated": TODAY, "notebook": "analysis/notebooks/05_hclt.ipynb", "n": {"hclt_2023_2025": int(len(hs)), "hclt_2026": int(len(h26))}, "family_by_set": family_by_set, "family_lift": family_lift,
           "year_counts": {str(k): {c: (float(v) if c == "share_pct" else int(v)) for c, v in row.items() if pd.notna(v)} for k, row in t1.iterrows()}},
          open(VER / f"hclt_notebook_{TODAY}.json", "w"), ensure_ascii=False, indent=1); print("saved", VER / f"hclt_notebook_{TODAY}.json")'''),
])

# ──────────────────────────────────────────────────────────────────────────
# 06 arXiv 2026년 1~9월 보조 자료(D24): 월별 건수·LLM 에이전트 비율, 대상 언어, HCLT 2026과 나란히
# ──────────────────────────────────────────────────────────────────────────

nb07 = nb([
    ("md", """# 07. 'agent'의 인접 연어(collocation) — 어떤 말과 함께 쓰이는가, 세계 전체 vs 한국어·중국어·일본어 집합 (2023-01 ~ 2026-09)

- **목적**(D37): LLM 에이전트 논문에서 'agent'가 바로 앞·뒤에 어떤 말과 함께 쓰이는지(multi-agent, llm agent, embodied agent / agent system, agent framework)를 문서 빈도로 세고, 그 경향이 세계 전체와 대상 언어 집합(D36 라벨 반영) 사이에, 그리고 연도 사이에 어떻게 다른지 본다. 토픽 모델과 달리 결정론적이고 재실행해도 같다.
- **입력**: `analysis/output/cache/agent_colloc_2023_2026q3.jsonl`(12_agent_collocations.py: spaCy 표제어·합성어 단위, 정본 `verification/agent_colloc_extract_*.json`), `preprocessing/output/corpus_llm_agent_all_2023_2026q3.csv`(연도·대상 언어 열)
- **자리(slot)**: `앞 말` = 접두(multi-agent의 multi) + 바로 앞 수식어(llm, autonomous, llm-based) / `뒤 말` = 바로 다음 명사(system, framework) / `agentic 뒤 말`
- **주의**
  - 문서 빈도(document frequency): 한 논문에 여러 번 나와도 1. 분모는 집합의 논문 수(세계 20,107 / 언어 집합 크기)
  - 수식어가 될 수 있는 말(명사·형용사·분사·합성어)만 센다. 'we propose agents', 'agents increasingly'의 동사·부사는 연어가 아니다
  - 'agent-based'처럼 agent가 접두인 합성어, 'agentic'은 머리(head)가 아니다. agentic은 뒤 말만 따로 본다
  - 2026은 1~9월. 언어 집합은 작으므로(한국어 25, 일본어 18) 비율보다 편수를 함께 읽는다
"""),
    ("code", COMMON_SETUP),
    ("md", """## 1. 자료 읽기 — 연어 캐시와 말뭉치 결합

- 계산: 캐시의 논문별 연어를 자리별 집합으로 바꿔 긴 표(id, slot, word)로 만든다. `앞 말`은 prefix와 left1을 합친다
- 정본 대조: 캐시의 논문 수 = 추출 정본의 n_docs = 말뭉치 전체
"""),
    ("code", '''ext = json.loads(Path(sorted(glob.glob(str(VER / "agent_colloc_extract_*.json")))[-1]).read_text())
recs = [json.loads(l) for l in open(CACHE / f"agent_colloc_{SUFFIX}.jsonl")]
assert len(recs) == ext["n_docs"] == len(allc), "캐시·정본·말뭉치의 논문 수가 다르다"
rows = []
for r in recs:
    for w in set(r["prefix"]) | set(r["left1"]): rows.append((r["id"], "before", w))
    for w in set(r["right1"]): rows.append((r["id"], "after", w))
    for w in set(r["agentic_right1"]): rows.append((r["id"], "agentic_after", w))
col = pd.DataFrame(rows, columns=["id", "slot", "word"]).drop_duplicates()
col = col.merge(allc[["id", "year"] + [f"lang_{l}" for l in LANGS]], on="id", how="left")
n_head = sum(r["n_head"] > 0 for r in recs)
N = {"world": len(allc), **{l: int(allc[f"lang_{l}"].sum()) for l in LANGS}}
print(f"논문 {len(allc):,}편 중 agent 머리 출현 {n_head:,}편({100*n_head/len(allc):.1f}%), 머리 출현 {ext['n_head_occurrences']:,}회 | 연어 행 {len(col):,} | 집합 크기 {N}")'''),
    ("md", """## 2. 세계 전체 — agent 앞 말·뒤 말 상위 30과 연도 추이

- 계산: 자리별 단어의 문서 빈도와 비율(분모 20,107), 연도별 비율(분모 그 해 논문 수), 네 점 직선 기울기(pp/년)
- 출력: `analysis/output/tables/agent_colloc_before_world.csv`, `agent_colloc_after_world.csv`, `agent_colloc_agentic_after_world.csv`
- 읽는 법: 'multi'는 multi-agent, 'llm'은 llm agent(s), 'llm-based'는 llm-based agent(s). 기울기 +면 그 쓰임이 늘고 있다
"""),
    ("code", '''n_year = allc.groupby("year").size().reindex(YEARS)
def world_table(slot, top=30):
    g = col[col.slot == slot]
    t = g.groupby("word").id.nunique().rename("n").to_frame(); t["pct"] = (100 * t.n / N["world"]).round(2)
    for y in YEARS:
        t[f"pct_{y}"] = (100 * g[g.year == y].groupby("word").id.nunique().reindex(t.index).fillna(0) / n_year[y]).round(2)
    t["slope_pp_per_year"] = [round(float(np.polyfit(YEARS, t.loc[w, [f"pct_{y}" for y in YEARS]].values.astype(float), 1)[0]), 2) for w in t.index]
    return t.sort_values("n", ascending=False).head(top)
W = {s: world_table(s) for s in ("before", "after", "agentic_after")}
for s, t in W.items(): t.to_csv(TAB / f"agent_colloc_{s}_world.csv")
print("agent 앞 말(세계, 문서 %):", ", ".join(f"{w}({r.pct:.1f}%, {r.slope_pp_per_year:+.1f})" for w, r in W["before"].head(20).iterrows()))
print("agent 뒤 말(세계, 문서 %):", ", ".join(f"{w}({r.pct:.1f}%, {r.slope_pp_per_year:+.1f})" for w, r in W["after"].head(20).iterrows()))
print("agentic 뒤 말(세계, 문서 %):", ", ".join(f"{w}({r.pct:.1f}%, {r.slope_pp_per_year:+.1f})" for w, r in W["agentic_after"].head(12).iterrows()))
W["before"]'''),
    ("md", """## 3. 세계 vs 한국어·중국어·일본어 집합 — 같은 단어의 관측 편수와 기대 편수

- 계산: 세계 상위 단어마다 언어 집합의 관측 편수(문서 빈도)와 기대 편수(세계 비율 × 집합 크기). 관측/기대가 1.5 초과면 많음, 0.5 미만이면 적음(7b절 LDA 표와 같은 기준). 집합이 작으므로 기대 1 미만인 칸은 판정하지 않는다
- 그리고 언어 집합에만 두드러지는 단어: 집합 안 3편 이상이고 세계 비율의 3배 이상
- 출력: `analysis/output/tables/agent_colloc_before_lang.csv`, `agent_colloc_after_lang.csv`, `agent_colloc_lang_specific.csv`, `analysis/output/figures/fig8a_agent_colloc_before_lang.png`
"""),
    ("code", '''def lang_table(slot, top=30):
    t = W[slot].copy()[["n", "pct"]].rename(columns={"n": "world_n", "pct": "world_pct"})
    g = col[col.slot == slot]
    for l in LANGS:
        obs = g[g[f"lang_{l}"]].groupby("word").id.nunique().reindex(t.index).fillna(0).astype(int)
        t[f"{l}_obs"] = obs; t[f"{l}_pct"] = (100 * obs / N[l]).round(1); t[f"{l}_exp"] = (t.world_pct / 100 * N[l]).round(1)
        ratio = obs / t[f"{l}_exp"].replace(0, np.nan)
        t[f"{l}_gap"] = np.where(t[f"{l}_exp"] < 1, "n/a", np.where(ratio > 1.5, "over", np.where(ratio < 0.5, "under", "near")))
    return t
L = {s: lang_table(s) for s in ("before", "after")}
for s, t in L.items(): t.to_csv(TAB / f"agent_colloc_{s}_lang.csv")
def md_cell(r, l): return f"{int(r[f'{l}_obs'])} ({r[f'{l}_pct']:.0f}%) · 기대 {r[f'{l}_exp']:.1f}" + {"over": " · 많음", "under": " · 적음"}.get(r[f"{l}_gap"], "")
md = f"| 자리 | 단어 | 세계 (n={N['world']:,}) | " + " | ".join(f"{LANG_NAME[l]} (n={N[l]})" for l in LANGS) + " |\\n|---|---|---|---|---|---|\\n"
for s, name in (("before", "agent 앞"), ("after", "agent 뒤")):
    for w, r in L[s].head(15).iterrows():
        md += f"| {name} | {w} | {int(r.world_n):,} ({r.world_pct:.1f}%) | " + " | ".join(md_cell(r, l) for l in LANGS) + " |\\n"
(TAB / "agent_colloc_world_vs_lang.md").write_text("문서 빈도(논문 수)와 집합 내 비율. 기대 = 세계 비율 × 집합 크기. 많음 = 기대의 1.5배 초과, 적음 = 절반 미만(기대 1 미만은 판정 안 함).\\n\\n" + md)
# 언어 집합에만 두드러지는 단어
spec_rows = []
for s in ("before", "after"):
    g = col[col.slot == s]; wp = (g.groupby("word").id.nunique() / N["world"])
    for l in LANGS:
        lp = g[g[f"lang_{l}"]].groupby("word").id.nunique()
        for w, n in lp[lp >= 3].items():
            if n / N[l] >= 3 * wp.get(w, 0):
                spec_rows.append({"slot": s, "lang": LANG_NAME[l], "word": w, "lang_n": int(n), "lang_pct": round(100 * n / N[l], 1), "world_pct": round(100 * wp.get(w, 0), 2)})
spec = pd.DataFrame(spec_rows).sort_values(["lang", "slot", "lang_n"], ascending=[True, True, False]); spec.to_csv(TAB / "agent_colloc_lang_specific.csv", index=False)
print("언어 집합에 두드러지는 연어(3편 이상, 세계 비율의 3배 이상):"); print(spec.to_string(index=False) if len(spec) else "(없음)")
display(L["before"].head(15))
fig, ax = plt.subplots(figsize=(8, 5.5)); t = L["before"].head(20); y = np.arange(len(t))
ax.barh(y, t.world_pct, color="#B0B7C3", label=f"world (n={N['world']:,})")
for l, mk, c in zip(LANGS, ["o", "s", "^"], ["#C0392B", "#2E86C1", "#27AE60"]): ax.scatter(t[f"{l}_pct"], y, marker=mk, color=c, label=f"{LANG_NAME[l]} (n={N[l]})", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(t.index); ax.invert_yaxis(); ax.set_xlabel("% of papers in the set (document frequency)"); ax.set_title("Words immediately before 'agent(s)': world vs target-language sets"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig8a_agent_colloc_before_lang.png", dpi=160); plt.show()'''),
    ("md", """## 3b. 언어 집합 칸의 이항 검정과 llm 계열 합산 (리뷰 2026-10-14)

- 계산: 세계 비율을 귀무 비율로 두고 집합 관측 편수의 양측 이항 검정 p(보정 없음, 기술 통계). 앞 말·뒤 말 세계 상위 30 × 언어 3 = 180칸 + 언어 특이어.
- llm 계열 합산: 앞 말이 llm, llm-* (llm-based, llm-driven, llm-powered 등), 'language model', 'language model-based' 가운데 하나라도 있는 논문 수(논문 단위 합집합)와 세계 비율 기준 기대·이항 p.
- 출력: `analysis/output/tables/agent_colloc_binomial.csv`
"""),
    ("code", '''from scipy.stats import binomtest
rows_b = []
for s_ in ("before", "after"):
    g = col[col.slot == s_]; wp = g.groupby("word").id.nunique() / N["world"]
    words_b = set(W[s_].index) | set(spec[spec.slot == s_].word)
    for w in words_b:
        for l in LANGS:
            n_l = int(g[g[f"lang_{l}"] & (g.word == w)].id.nunique()); p0 = float(wp.get(w, 0))
            rows_b.append({"slot": s_, "word": w, "lang": l, "obs": n_l, "n": N[l], "world_pct": round(100 * p0, 2), "exp": round(p0 * N[l], 1), "lang_pct": round(100 * n_l / N[l], 1), "p_binom": round(binomtest(n_l, N[l], p0).pvalue, 4) if p0 > 0 else None})
BT = pd.DataFrame(rows_b); BT.to_csv(TAB / "agent_colloc_binomial.csv", index=False)
sigB = BT[(BT.p_binom.notna()) & (BT.p_binom < 0.05)].sort_values(["lang", "slot", "p_binom"])
print("이항 검정 p<0.05인 칸:"); print(sigB[["lang", "slot", "word", "obs", "n", "exp", "world_pct", "lang_pct", "p_binom"]].to_string(index=False))
# llm 계열 합산(논문 단위)
LLM_RE_W = re.compile(r"^(llm|llm-.*|language model|language model-based|large language model)$")
llm_fam = col[(col.slot == "before") & col.word.str.match(LLM_RE_W)]
llm_union = {"world": int(llm_fam.id.nunique())}
for l in LANGS: llm_union[l] = int(llm_fam[llm_fam[f"lang_{l}"]].id.nunique())
p0 = llm_union["world"] / N["world"]
llm_union_stats = {l: {"obs": llm_union[l], "exp": round(p0 * N[l], 1), "p_binom": round(binomtest(llm_union[l], N[l], p0).pvalue, 4)} for l in LANGS}
print(f"\\nllm 계열 합산: 세계 {llm_union['world']:,}편({100*p0:.1f}%)", llm_union_stats)'''),
    ("md", """## 4. 연도 추이 — 세계 상위 앞 말·뒤 말의 비율 변화

- 계산: 2절의 연도별 비율을 선으로. 분모는 그 해 논문 수(2026은 1~9월)
- 출력: `analysis/output/figures/fig8b_agent_colloc_trend.png`
"""),
    ("code", '''fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, s, title in zip(axes, ("before", "after"), ("before 'agent(s)'", "after 'agent(s)'")):
    t = W[s].head(10)
    for w, r in t.iterrows(): ax.plot(YEARS, [r[f"pct_{y}"] for y in YEARS], marker="o", label=w)
    ax.set_xticks(YEARS); ax.set_xticklabels([str(y) if y != 2026 else "2026\\n(Jan-Sep)" for y in YEARS]); ax.set_ylabel("% of papers that year"); ax.set_title(f"Top 10 words {title}"); ax.legend(fontsize=7, frameon=False, ncol=2)
plt.tight_layout(); plt.savefig(FIG / "fig8b_agent_colloc_trend.png", dpi=160); plt.show()
rising = W["before"][W["before"].slope_pp_per_year >= 0.5].index.tolist(); falling = W["before"][W["before"].slope_pp_per_year <= -0.5].index.tolist()
print("앞 말 오름(+0.5pp/년 이상):", rising); print("앞 말 내림(−0.5pp/년 이하):", falling)'''),
    ("md", """## 5. 정본 저장

- 저장: 세계 상위 표, 언어 집합 관측·기대, 언어 특이 연어, 집합 크기 → `verification/agent_colloc_YYYYMMDD.json`
"""),
    ("code", '''out = {"generated": TODAY, "notebook": "analysis/notebooks/07_agent_collocations.ipynb", "period": counts["definitions"]["period"], "extract": ext,
       "set_sizes": N, "n_docs_with_agent_head": int(n_head),
       "world": {s: {w: {c: float(v) for c, v in r.items()} for w, r in t.iterrows()} for s, t in W.items()},
       "lang": {s: {w: {c: (float(v) if not isinstance(v, str) else v) for c, v in r.items()} for w, r in t.iterrows()} for s, t in L.items()},
       "lang_specific": spec.to_dict(orient="records"), "binomial_sig": sigB.to_dict(orient="records"), "llm_family_union": {"world": llm_union["world"], "world_pct": round(100 * p0, 2), **llm_union_stats}}
p = VER / f"agent_colloc_{TODAY}.json"; p.write_text(json.dumps(out, ensure_ascii=False, indent=1)); print("saved", p)'''),
])

META_SETUP = '''# 메타분석 공통: spaCy 표제어 캐시(02에서 만든 것)와 LDA 문서 배정을 읽는다
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
DOMAIN_STOP = {"paper", "propose", "proposed", "method", "methods", "result", "results", "show", "shows", "model", "models", "based", "approach", "approaches",
               "task", "tasks", "performance", "llm", "llms", "agent", "agents", "large", "language", "framework", "novel", "existing", "demonstrate",
               "experiments", "experimental", "study", "work", "using", "use", "data", "code", "available", "https", "github", "com", "et", "al"}
STOP = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP
lem = {}
for line in open(CACHE / f"lemmas_{SUFFIX}.jsonl"):
    r = json.loads(line); lem[r["id"]] = [t for t in r["t"] if t not in STOP and len(t) > 2 and not t.isdigit()]
assert set(lem) == set(allc.id), "표제어 캐시와 말뭉치의 id가 다르다"
dt = pd.read_csv(TAB / "lda_doc_topics.csv", dtype={"id": str}); assert len(dt) == len(allc)
allc = allc.merge(dt[["id", "lda_topic", "lda_prob"]], on="id", how="left")
ld = json.loads(Path(sorted(glob.glob(str(VER / "topics_lda_*.json")))[-1]).read_text()); K = ld["K"]
label = {int(t): v["label"] for t, v in ld["topics"].items()}
allc["ym"] = allc.year * 100 + allc.month
MONTHS = sorted(allc.ym.unique()); assert len(MONTHS) == 45 and MONTHS[0] == 202301 and MONTHS[-1] == 202609
print("문서", len(allc), "| 표제어 토큰", sum(len(v) for v in lem.values()), "| 토픽", K, "| 월", len(MONTHS))'''

nb08 = nb([
    ("md", """# 08. 메타분석 1 — 키니스(keyness): 언어 집합과 2026년에 유의하게 더 쓰이는 말 (로그 오즈비, Dirichlet 사전분포)

- **목적**: 어휘 전체를 대상으로 "이 집합에서 더 쓰이는 말"을 순위로 낸다. Monroe·Colaresi·Quinn(2008)의 정보적 Dirichlet 사전분포 로그 오즈비(log-odds ratio, informative Dirichlet prior)와 z 점수. 연어 분석(07)이 'agent' 주변만 본다면 키니스는 어휘 전체를 본다.
- **비교**: (가) 한국어·중국어·일본어 집합(사람 라벨, D36) 대 나머지 세계, (나) 2026년(1~9월) 대 2023~2025년 세계 전체.
- **입력**: spaCy 표제어 캐시 `analysis/output/cache/lemmas_2023_2026q3.jsonl`(제목+초록, 02와 같은 전처리), `allc`
- **계산**: 집합 i와 j의 단어 빈도 $y_{iw}, y_{jw}$, 사전분포 $\\alpha_w = \\alpha_0 \\cdot n_w / N$(세계 전체 빈도 비례, $\\alpha_0 = 1{,}000$). $\\delta_w = \\log\\frac{y_{iw}+\\alpha_w}{n_i+\\alpha_0-y_{iw}-\\alpha_w} - \\log\\frac{y_{jw}+\\alpha_w}{n_j+\\alpha_0-y_{jw}-\\alpha_w}$, $z_w = \\delta_w / \\sqrt{1/(y_{iw}+\\alpha_w) + 1/(y_{jw}+\\alpha_w)}$. 전체 빈도 20 미만인 단어는 뺀다.
- **주의**: 토큰 빈도 기준이라 한 논문에서 반복된 말이 크게 잡힌다(문서 빈도도 함께 적는다). 언어 이름(korean, chinese, japanese)은 집합 정의에 쓰였으므로 결과에서 뺀다. 작은 집합(한국어 25, 일본어 18)은 z가 작아도 순위는 읽을 수 있다.
"""),
    ("code", COMMON_SETUP),
    ("code", META_SETUP),
    ("md", """## 1. 로그 오즈비 계산 함수와 집합별 빈도"""),
    ("code", '''LANG_WORDS = {"korean", "korea", "chinese", "china", "mandarin", "japanese", "japan", "hangul", "cjk", "english"}
def counts_of(ids):
    c = Counter(); d = Counter()
    for i in ids:
        toks = lem[i]; c.update(toks); d.update(set(toks))
    return c, d
world_c, world_d = counts_of(allc.id); N = sum(world_c.values())
vocab = [w for w, n in world_c.items() if n >= 20 and w not in LANG_WORDS]; print("어휘(빈도 20 이상)", len(vocab))
def log_odds(ci, ni, cj, nj, alpha0=1000.0):
    rows = []
    for w in vocab:
        aw = alpha0 * world_c[w] / N; yi, yj = ci.get(w, 0), cj.get(w, 0)
        d = np.log((yi + aw) / (ni + alpha0 - yi - aw)) - np.log((yj + aw) / (nj + alpha0 - yj - aw))
        z = d / np.sqrt(1 / (yi + aw) + 1 / (yj + aw)); rows.append((w, yi, yj, round(float(d), 3), round(float(z), 2)))
    return pd.DataFrame(rows, columns=["word", "n_in", "n_rest", "log_odds", "z"]).set_index("word")'''),
    ("md", """## 2. 언어 집합 대 나머지 세계 — 상위 20(많이 쓰는 말)과 하위 10(적게 쓰는 말)

- 출력: `analysis/output/tables/keyness_lang_{ko,zh,ja}.csv`, `analysis/output/figures/fig9a_keyness_lang.png`
- 읽는 법: z가 클수록 그 집합에서 상대적으로 많이 쓰인 말. 문서 빈도(df_in)가 2~3편이면 특정 논문의 반복일 수 있다.
"""),
    ("code", '''KL = {}
for l in LANGS:
    ids_in = allc[allc[f"lang_{l}"]].id; ids_out = allc[~allc[f"lang_{l}"]].id
    ci, di = counts_of(ids_in); cj, dj = counts_of(ids_out)
    t = log_odds(ci, sum(ci.values()), cj, sum(cj.values())); t["df_in"] = [di.get(w, 0) for w in t.index]; t["df_rest"] = [dj.get(w, 0) for w in t.index]
    t["pct_docs_in"] = (100 * t.df_in / len(ids_in)).round(1); t["pct_docs_rest"] = (100 * t.df_rest / len(ids_out)).round(2)
    t = t.sort_values("z", ascending=False); t.to_csv(TAB / f"keyness_lang_{l}.csv"); KL[l] = t
    print(f"\\n{LANG_NAME[l]} (n={len(ids_in)}) 많이 쓰는 말:", ", ".join(f"{w}(z={r.z:.1f}, {int(r.df_in)}편)" for w, r in t.head(20).iterrows()))
    print(f"{LANG_NAME[l]} 적게 쓰는 말:", ", ".join(f"{w}(z={r.z:.1f})" for w, r in t.tail(10).iloc[::-1].iterrows()))
fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
for ax, l in zip(axes, LANGS):
    t = KL[l].head(20).iloc[::-1]; ax.barh(t.index, t.z, color="#2E86C1"); ax.set_title(f"{LANG_NAME[l]} vs rest (n={int(allc[f'lang_{l}'].sum())})", fontsize=10); ax.set_xlabel("z (log-odds, Dirichlet prior)")
plt.suptitle("Keyness: words over-used in each target-language set (top 20 by z)", fontsize=11); plt.tight_layout(); plt.savefig(FIG / "fig9a_keyness_lang.png", dpi=160); plt.show()'''),
    ("md", """## 3. 2026년 대 2023~2025년 — 세계 전체에서 새로 늘어난 말

- 출력: `analysis/output/tables/keyness_2026_vs_prior.csv`
- 읽는 법: 2026년 논문에서 상대적으로 많이 쓰인 말(상위)과 줄어든 말(하위). 07의 연어, 09의 버스트와 겹치는지 본다.
"""),
    ("code", '''ids_26 = allc[allc.year == 2026].id; ids_pr = allc[allc.year < 2026].id
c26, d26 = counts_of(ids_26); cpr, dpr = counts_of(ids_pr)
t26 = log_odds(c26, sum(c26.values()), cpr, sum(cpr.values())); t26["df_2026"] = [d26.get(w, 0) for w in t26.index]; t26["df_prior"] = [dpr.get(w, 0) for w in t26.index]
t26["pct_docs_2026"] = (100 * t26.df_2026 / len(ids_26)).round(2); t26["pct_docs_prior"] = (100 * t26.df_prior / len(ids_pr)).round(2)
t26 = t26.sort_values("z", ascending=False); t26.to_csv(TAB / "keyness_2026_vs_prior.csv")
print("2026년에 늘어난 말:", ", ".join(f"{w}(z={r.z:.0f}, {r.pct_docs_2026:.1f}%←{r.pct_docs_prior:.1f}%)" for w, r in t26.head(30).iterrows()))
print("\\n2026년에 줄어든 말:", ", ".join(f"{w}(z={r.z:.0f}, {r.pct_docs_2026:.1f}%←{r.pct_docs_prior:.1f}%)" for w, r in t26.tail(20).iloc[::-1].iterrows()))'''),
    ("md", """## 4. 가중 키워드 동시출현 연결망(co-word network) — 전체 기간과 2023~2025 대 2026

- 계산: 표제어(불용어·도메인 일반어·언어 이름 제외)와 인접 표제어 2-gram 가운데 문서 빈도 상위 60개를 노드로 둔다. 간선 가중치 = 두 키워드가 같은 논문에 함께 나타난 논문 수, 가중치 상위 250개 간선만 그린다. 노드 크기 ∝ 문서 빈도. 참고 논문(류지혜 외 2025)의 그림 2와 같은 구성이며, 옛 노트북 04의 정규식 토큰 판을 spaCy 표제어·2023-01~2026-09 LLM 에이전트 20,107편으로 바꾼 것이다.
- 배치는 spring layout(seed 42)이며 거리에 뜻을 두지 않는다. 전체 기간 한 장, 2023~2025와 2026(1~9월)을 나란히 한 장.
- 출력: `analysis/output/tables/coword_top60_{all,2023_2025,2026}.csv`, `analysis/output/figures/fig9e_coword_network_all.png`, `fig9f_coword_network_2023_2025_vs_2026.png`
"""),
    ("code", '''import networkx as nx
GENERIC = {"introduce", "present", "presents", "significant", "potential", "including", "achieves", "achieve", "extensive", "current", "provide",
           "provides", "evaluate", "effectiveness", "efficiency", "key", "critical", "challenge", "challenges", "challenging", "dynamic",
           "efficient", "compared", "effective", "time", "robust", "recent", "multiple", "diverse", "process", "problem", "problems", "research",
           "information", "complex", "state-of-the-art", "analysis", "applications", "application", "important", "enable", "enables", "leverage",
           "leverages", "improve", "improves", "improved", "outperforms", "existing", "propose", "proposes", "paper", "design", "designed",
           "develop", "developed", "ability", "capabilities", "capability", "scenarios", "real-world", "settings", "setting", "wide", "range",
           "used", "enabling", "future", "limited", "comprehensive", "effectively", "specifically", "remains", "need", "furthermore", "like",
           "traditional", "development", "limitations", "enhance", "generate", "learn", "space", "context", "features", "network", "networks",
           "algorithm", "algorithms", "neural", "deep", "strategies", "understanding", "interactions", "interaction", "actions", "behavior",
           "control", "optimization", "detection", "accuracy", "dataset", "datasets", "benchmark", "benchmarks", "evaluation", "generation"}
NET_STOP = STOP | LANG_WORDS | GENERIC | {"require", "address", "remain", "high", "new", "different", "various", "specific", "well", "work", "approach", "exist", "include", "produce", "reduce", "support", "increasingly", "strong", "single", "rate", "gap", "ground", "level", "real", "offer", "drive", "datum", "experiment", "result", "show", "find", "highlight", "demonstrate", "compare", "base", "ensure", "allow", "make", "large", "small", "low", "open", "long", "source", "quality", "accuracy", "generate", "generation", "build", "context", "domain", "world", "scale", "general", "novel", "overall", "finally", "additionally", "study"}
def doc_units(toks):
    u = set(w for w in toks if w not in NET_STOP)
    u |= set(f"{a}_{b}" for a, b in zip(toks, toks[1:]) if a not in NET_STOP and b not in NET_STOP)
    return u
def coword(ids, top=60, edges=250):
    units = {i: doc_units(lem[i]) for i in ids}
    dfc = Counter(); [dfc.update(u) for u in units.values()]
    # 2-gram이 상위에 들면 그 구성 단어 하나짜리는 뺀다(중복 노드 억제)
    keep = []
    for w, n in dfc.most_common(400):
        if "_" in w and all(p not in keep for p in w.split("_")) or "_" not in w and not any(w in k.split("_") for k in keep):
            keep.append(w)
        if len(keep) == top: break
    kset = set(keep); co = Counter()
    for u in units.values():
        ks = sorted(u & kset)
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)): co[(ks[a], ks[b])] += 1
    G = nx.Graph(); [G.add_node(w, df=dfc[w]) for w in keep]
    for (a, b), n in co.most_common(edges): G.add_edge(a, b, weight=n)
    return G, dfc
def draw(G, ax, title):
    G = G.subgraph([n for n in G if G.degree(n) > 0]).copy()                      # 상위 250 간선에 들지 못한 고립 노드는 그리지 않는다
    wmax = max(d["weight"] for _, _, d in G.edges(data=True))
    for a, b, d in G.edges(data=True): d["w01"] = d["weight"] / wmax                 # 끌림 세기는 0~1로 정규화(원 가중치는 수천 편이라 배치가 뭉친다)
    pos = nx.spring_layout(G, seed=42, k=1.6 / np.sqrt(G.number_of_nodes()), weight="w01", iterations=200); sizes = [G.nodes[w]["df"] for w in G]; mx = max(sizes)
    nx.draw_networkx_edges(G, pos, ax=ax, width=[0.3 + 3 * G[a][b]["w01"] for a, b in G.edges()], edge_color="#999999", alpha=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[300 + 2500 * s / mx for s in sizes], node_color="#48C9B0", alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold"); ax.set_title(title, fontsize=10); ax.axis("off")
sets_net = {"all": allc.id, "2023_2025": allc[allc.year <= 2025].id, "2026": allc[allc.year == 2026].id}
Gs = {}
for name, ids in sets_net.items():
    G, dfc = coword(ids); Gs[name] = G
    pd.DataFrame({"keyword": list(G.nodes), "doc_freq": [G.nodes[w]["df"] for w in G.nodes], "doc_pct": [round(100 * G.nodes[w]["df"] / len(ids), 1) for w in G.nodes]}).sort_values("doc_freq", ascending=False).to_csv(TAB / f"coword_top60_{name}.csv", index=False)
    print(name, f"n={len(ids):,}", "| 상위 15:", ", ".join(f"{w}({G.nodes[w]['df']})" for w in sorted(G.nodes, key=lambda w: -G.nodes[w]["df"])[:15]))
fig, ax = plt.subplots(figsize=(13, 10)); draw(Gs["all"], ax, f"Weighted keyword co-occurrence network, LLM-agent papers 2023-01 to 2026-09 (n={len(allc):,}; top-60 keywords, top-250 edges)")
plt.tight_layout(); plt.savefig(FIG / "fig9e_coword_network_all.png", dpi=150, bbox_inches="tight"); plt.show()
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
draw(Gs["2023_2025"], axes[0], f"2023-2025 (n={len(sets_net['2023_2025']):,})"); draw(Gs["2026"], axes[1], f"2026 Jan-Sep (n={len(sets_net['2026']):,})")
plt.tight_layout(); plt.savefig(FIG / "fig9f_coword_network_2023_2025_vs_2026.png", dpi=150, bbox_inches="tight"); plt.show()
only26 = sorted(set(Gs["2026"].nodes) - set(Gs["2023_2025"].nodes)); only_pre = sorted(set(Gs["2023_2025"].nodes) - set(Gs["2026"].nodes))
print("2026 연결망에만 있는 노드:", only26); print("2023~2025 연결망에만 있는 노드:", only_pre)
coword_summary = {"top60": {k: [{"keyword": w, "df": int(G.nodes[w]["df"])} for w in sorted(G.nodes, key=lambda w: -G.nodes[w]["df"])] for k, G in Gs.items()}, "only_2026": only26, "only_2023_2025": only_pre, "edges": 250}'''),
    ("md", """## 5. 정본 저장"""),
    ("code", '''out = {"generated": TODAY, "notebook": "analysis/notebooks/08_keyness.ipynb", "coword_network": coword_summary, "method": "log-odds ratio with informative Dirichlet prior (Monroe et al. 2008), alpha0=1000, vocab min count 20, language names excluded",
       "n_vocab": len(vocab), "set_sizes": {l: int(allc[f"lang_{l}"].sum()) for l in LANGS}, "n_2026": int(len(ids_26)), "n_prior": int(len(ids_pr)),
       "lang_top": {l: KL[l].head(25).reset_index().to_dict(orient="records") for l in LANGS}, "lang_bottom": {l: KL[l].tail(10).iloc[::-1].reset_index().to_dict(orient="records") for l in LANGS},
       "y2026_top": t26.head(40).reset_index().to_dict(orient="records"), "y2026_bottom": t26.tail(25).iloc[::-1].reset_index().to_dict(orient="records")}
p = VER / f"meta_keyness_{TODAY}.json"; p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float)); print("saved", p)'''),
])


nb10 = nb([
    ("md", """# 10. 메타분석 3 — 토픽 비중의 공변량 분석(STM 대체): 연도를 조정한 기대 편수와 로지스틱 회귀

- **목적**: 언어 집합이 2026년에 치우쳐 있으므로(한국어 25편 중 14편이 2026), 토픽별 "기대 편수"를 연도 구성에 맞춰 조정하고도 관측·기대 차이가 남는지 본다. 구조적 토픽 모델(STM)이 공변량으로 토픽 비중을 설명하는 것과 같은 질문을, 이미 배정된 LDA 토픽에 대해 회귀로 답한다. R의 stm이 설치되면 같은 입력으로 STM을 돌려 비교한다.
- **입력**: `lda_doc_topics.csv`(02의 문서 배정), `allc`(연도, 언어 집합)
- **계산**: (가) 연도 조정 기대 편수 $E_{adj}(t,l) = \\sum_y \\pi_{t,y} \\cdot n_{l,y}$($\\pi_{t,y}$ = 그 해 세계의 토픽 $t$ 비중, $n_{l,y}$ = 집합 $l$의 그 해 편수). (나) 토픽마다 로지스틱 회귀 $\\mathrm{logit}\\,P(\\text{topic}=t) = \\beta_0 + \\beta_{ko} x_{ko} + \\beta_{zh} x_{zh} + \\beta_{ja} x_{ja} + \\sum_y \\beta_y \\mathbb{1}[\\text{year}=y]$, 전체 20,107편. 오즈비 $e^{\\beta_l}$과 Wald 95% 신뢰구간. 집합 안 관측이 0편인 칸은 분리(separation)라 추정하지 않는다.
- **주의**: 토픽은 확률이 가장 큰 하나에 배정한 값이라 혼합 비중이 아니다. 신뢰구간은 작은 집합에서 넓다. 1위 토픽 benchmark·score는 2026년 문체 토픽이라 해석에서 뺀다(3.2절).
"""),
    ("code", COMMON_SETUP),
    ("code", META_SETUP),
    ("md", """## 1. 연도 조정 기대 편수 대 단순 기대 편수"""),
    ("code", '''import statsmodels.api as sm
pi = pd.crosstab(allc.lda_topic, allc.year, normalize="columns")       # 토픽 × 연도 비중(열 합 1)
rows = []
for t in range(K):
    r = {"topic": t, "label": label[t], "world_n": int((allc.lda_topic == t).sum())}
    for l in LANGS:
        sub = allc[allc[f"lang_{l}"]]; n_ly = sub.groupby("year").size().reindex(YEARS).fillna(0)
        obs = int((sub.lda_topic == t).sum()); exp_naive = float((allc.lda_topic == t).mean() * len(sub)); exp_adj = float(sum(pi.loc[t, y] * n_ly[y] for y in YEARS if y in pi.columns))
        r.update({f"{l}_obs": obs, f"{l}_exp_naive": round(exp_naive, 2), f"{l}_exp_adj": round(exp_adj, 2), f"{l}_ratio_adj": round(obs / exp_adj, 2) if exp_adj > 0 else None})
    rows.append(r)
A = pd.DataFrame(rows).set_index("topic"); A.to_csv(TAB / "topic_lang_year_adjusted.csv")
chg = A[[c for c in A.columns if c.endswith("_exp_naive") or c.endswith("_exp_adj")]]
print("연도 조정으로 기대 편수가 1편 이상 움직인 칸:")
for l in LANGS:
    d = (A[f"{l}_exp_adj"] - A[f"{l}_exp_naive"]); big = d[d.abs() >= 1].sort_values()
    print(" ", LANG_NAME[l], ", ".join(f"{label[t]} {A.loc[t, f'{l}_exp_naive']:.1f}→{A.loc[t, f'{l}_exp_adj']:.1f}" for t in big.index))'''),
    ("md", """## 2. 토픽별 로지스틱 회귀 — 언어 집합의 오즈비(연도 조정)

- 출력: `analysis/output/tables/topic_lang_logit.csv`, `analysis/output/figures/fig9c_topic_lang_or.png`
- 읽는 법: 오즈비 > 1이면 같은 해의 세계 논문보다 그 토픽에 들 확률이 높다. 신뢰구간이 1을 포함하지 않는 칸만 굵게 본다.
"""),
    ("code", '''X = pd.DataFrame({f"lang_{l}": allc[f"lang_{l}"].astype(float) for l in LANGS})
for y in YEARS[1:]: X[f"y{y}"] = (allc.year == y).astype(float)
X = sm.add_constant(X)
rows = []
for t in range(K):
    yv = (allc.lda_topic == t).astype(float)
    try:
        m = sm.Logit(yv, X).fit(disp=0, maxiter=200)
        for l in LANGS:
            obs = int(((allc.lda_topic == t) & allc[f"lang_{l}"]).sum())
            if obs == 0:
                rows.append({"topic": t, "label": label[t], "lang": l, "obs": 0, "OR": None, "ci_low": None, "ci_high": None, "p": None, "note": "obs 0 (separation)"}); continue
            b, se = m.params[f"lang_{l}"], m.bse[f"lang_{l}"]
            rows.append({"topic": t, "label": label[t], "lang": l, "obs": obs, "OR": round(float(np.exp(b)), 2), "ci_low": round(float(np.exp(b - 1.96 * se)), 2), "ci_high": round(float(np.exp(b + 1.96 * se)), 2), "p": round(float(m.pvalues[f"lang_{l}"]), 4), "note": ""})
    except Exception as e:
        for l in LANGS: rows.append({"topic": t, "label": label[t], "lang": l, "obs": None, "OR": None, "ci_low": None, "ci_high": None, "p": None, "note": f"fit failed: {e}"[:60]})
G = pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests
mask = G.p.notna(); G.loc[mask, "q_bh"] = multipletests(G.loc[mask, "p"].values, method="fdr_bh")[1].round(4)   # 다중 비교(47칸) BH 보정(리뷰 2026-10-14)
G.to_csv(TAB / "topic_lang_logit.csv", index=False)
sig = G[(G.OR.notna()) & ((G.ci_low > 1) | (G.ci_high < 1))].sort_values(["lang", "OR"], ascending=[True, False])
print("BH 보정 q<0.05인 칸:", ", ".join(f"{r.lang}:{r.label}({'+' if r.OR > 1 else '-'}, q={r.q_bh})" for r in G[G.q_bh < 0.05].itertuples()))
print("신뢰구간이 1을 포함하지 않는 칸(연도 조정 뒤):"); print(sig[["lang", "label", "obs", "OR", "ci_low", "ci_high", "p"]].to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharey=True)
for ax, l in zip(axes, LANGS):
    g = G[(G.lang == l) & G.OR.notna()].sort_values("topic"); y = np.arange(len(g))
    ax.errorbar(g.OR, y, xerr=[g.OR - g.ci_low, g.ci_high - g.OR], fmt="o", color=["#C0392B" if (lo > 1 or hi < 1) else "#7F8C8D" for lo, hi in zip(g.ci_low, g.ci_high)][0] if False else "#2E86C1", ecolor="#B0B7C3", capsize=2, ms=4)
    for i, (lo, hi, orv) in enumerate(zip(g.ci_low, g.ci_high, g.OR)):
        if lo > 1 or hi < 1: ax.plot(orv, i, "o", color="#C0392B", ms=5)
    ax.axvline(1, color="k", lw=0.6); ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([f"{t} {label[t]}" for t in g.topic], fontsize=7); ax.set_title(f"{LANG_NAME[l]} (n={int(allc[f'lang_{l}'].sum())})", fontsize=10); ax.set_xlabel("odds ratio (year-adjusted), log scale")
plt.suptitle("Topic membership odds ratios for target-language sets vs world, adjusted for year (red = 95% CI excludes 1)", fontsize=10); plt.tight_layout(); plt.savefig(FIG / "fig9c_topic_lang_or.png", dpi=160); plt.show()'''),
    ("md", """## 3. STM(구조적 토픽 모델, R stm) 결과와의 비교

- 입력: `verification/stm_*.json`(16_stm.R: K=25, prevalence ~ 언어 집합 + 연도, 스펙트럴 초기화). 없으면 이 절은 건너뛴다.
- 계산: (가) STM 토픽을 LDA 토픽에 상위 10단어 자카드 유사도로 대응시킨다(일대일이 아니므로 가장 가까운 것). (나) STM의 언어 집합 효과(estimateEffect, 연도 조정)가 유의한(p<0.05) 토픽과 2절의 로지스틱 회귀가 유의한 토픽을 나란히 둔다. (다) 언어 집합의 평균 토픽 비중(theta) 대 세계 비율.
- 주의: STM은 문서를 토픽 혼합으로 보고(theta), 2절은 최대 확률 토픽 하나로 본다. 두 모델의 토픽 경계가 달라 대응은 참고용이다.
"""),
    ("code", '''stm_files = sorted(glob.glob(str(VER / "stm_2*.json")))
if stm_files:
    st = json.loads(Path(stm_files[-1]).read_text())
    assert st["n_docs"] == len(allc) and st["vocab_size"] == ld["vocab_size"], "STM 입력이 현재 LDA와 다르다"
    assert st.get("input", {}).get("set_sizes") == {f"lang_{l}": int(allc[f"lang_{l}"].sum()) for l in LANGS}, "STM 언어 집합이 이전 버전이다. 15·16을 재실행한다."
    import hashlib
    assert st["input"].get("corpus_sha256") == hashlib.sha256((DATA / f"corpus_llm_agent_all_{SUFFIX}.csv").read_bytes()).hexdigest(), "STM 말뭉치가 현재 입력과 다르다"
    print("STM:", st["r_version"], "stm", st["stm_version"], "| 수렴", st["converged"], "| EM 반복", st["em_iterations"], "|", st["elapsed_min"], "분")
    lda_words = {int(t): set(v["top_words"][:10]) for t, v in ld["topics"].items()}
    stm_topics = pd.DataFrame(st["topics"]); stm_topics["words"] = stm_topics.prob_words.str.split()
    def best_match(ws):
        ws = set(ws[:10]); sims = {t: len(ws & lw) / len(ws | lw) for t, lw in lda_words.items()}; t = max(sims, key=sims.get); return t, round(sims[t], 2)
    stm_topics[["lda_match", "jaccard"]] = pd.DataFrame([best_match(w) for w in stm_topics.words], index=stm_topics.index)
    stm_topics["lda_label"] = stm_topics.lda_match.map(label)
    eff = pd.DataFrame(st["effects_lang"]); eff["lang"] = eff.covariate.str.replace("lang_", "")
    tb = pd.DataFrame(st["theta_by_set"]).set_index("topic")
    stm_sig = eff[eff.p < 0.05].merge(stm_topics[["topic", "lda_match", "lda_label", "jaccard"]], on="topic")
    stm_sig["direction"] = np.where(stm_sig.estimate > 0, "+", "-")
    print("\\nSTM에서 언어 집합 효과가 유의한(p<0.05) 토픽:")
    print(stm_sig.sort_values(["lang", "estimate"], ascending=[True, False])[["lang", "topic", "label", "lda_label", "jaccard", "estimate", "p", "direction"]].to_string(index=False))
    print("\\n로지스틱 회귀(2절)에서 유의한 칸:", ", ".join(f"{LN}:{r.label}({'+' if r.OR > 1 else '-'})" for LN, r in [(r.lang, r) for r in sig.itertuples()]))
    stm_topics[["topic", "label", "lda_match", "lda_label", "jaccard", "mean_theta", "prob_words"]].to_csv(TAB / "stm_vs_lda_topics.csv", index=False)
    # 방향 일치: STM 유의 토픽을 LDA 대응 토픽으로 옮겨 2절의 유의 칸과 비교
    agree = []
    for r in stm_sig.itertuples():
        m = sig[(sig.lang == r.lang) & (sig.topic == r.lda_match)]
        agree.append({"lang": r.lang, "stm_topic": r.topic, "stm_label": r.label, "lda_label": r.lda_label, "jaccard": r.jaccard, "stm_dir": r.direction, "logit_dir": ("+" if m.OR.iloc[0] > 1 else "-") if len(m) else "n.s."})
    AG = pd.DataFrame(agree); print("\\nSTM 유의 토픽의 LDA 대응과 로지스틱 회귀 방향:"); print(AG.to_string(index=False) if len(AG) else "(없음)")
    stm_summary = {"file": Path(stm_files[-1]).name, "converged": st["converged"], "em_iterations": st["em_iterations"], "elapsed_min": st["elapsed_min"], "significant": stm_sig.to_dict(orient="records"), "agreement": AG.to_dict(orient="records"), "topic_match": stm_topics[["topic", "label", "lda_match", "lda_label", "jaccard", "mean_theta"]].to_dict(orient="records")}
else:
    stm_summary = None; print("STM 정본이 아직 없다(16_stm.R 실행 전).")'''),
    ("md", """## 4. 정본 저장"""),
    ("code", '''out = {"generated": TODAY, "notebook": "analysis/notebooks/10_topic_covariates.ipynb", "method": "year-adjusted expected counts; per-topic logistic regression with language indicators and year dummies (statsmodels Logit, Wald 95% CI)", "stm": stm_summary,
       "year_adjusted": {str(t): {c: (None if pd.isna(v) else (float(v) if not isinstance(v, str) else v)) for c, v in r.items()} for t, r in A.iterrows()},
       "logit": G.to_dict(orient="records"), "significant": sig.to_dict(orient="records"), "excluded_topic_note": "benchmark·score(2026 style topic) excluded from interpretation"}
p = VER / f"meta_topic_covariates_{TODAY}.json"; p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: None if pd.isna(x) else float(x))); print("saved", p)'''),
])


import sys
ALL = [("01_corpus_overview.ipynb", nb01), ("02_lda.ipynb", nb02),
       ("05_hclt.ipynb", nb05), ("07_agent_collocations.ipynb", nb07),
       ("08_keyness.ipynb", nb08), ("10_topic_covariates.ipynb", nb10)]

want = sys.argv[1:]            # 예: python3 scripts/02_make_notebooks.py 04  → 04만 다시 만든다(실행된 다른 노트북을 지우지 않는다)
for name, n in ALL:
    if want and not any(name.startswith(w) for w in want):
        continue
    nbf.write(n, OUT / name)
    print("wrote", OUT / name)
