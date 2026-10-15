"""장부(manifest)와 정본 JSON의 정합성 검사. 실행: python3 -m pytest tests"""
import glob
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def latest(pattern):
    return sorted(glob.glob(str(ROOT / pattern)))[-1]


def test_manifest_hashes_and_keys():
    m = json.loads((ROOT / "preprocessing/output/corpus_manifest.json").read_text())
    for f in m["files"]:
        p = ROOT / f["path"]
        assert sha256(p) == f["sha256"], f["path"]
        df = pd.read_csv(p, dtype={"id": str})
        assert len(df) == f["rows"]
        assert df["id"].is_unique


def test_counts_match_corpus():
    v = json.loads(Path(latest("verification/corpus_counts_*.json")).read_text())
    allc = pd.read_csv(ROOT / "preprocessing/output/corpus_llm_agent_all_2023_2026q3.csv", dtype={"id": str})
    cjk = pd.read_csv(ROOT / "preprocessing/output/corpus_llm_agent_cjk_lang_2023_2026q3.csv", dtype={"id": str})
    assert len(allc) == v["pipeline"]["llm_agent_all"]
    assert len(cjk) == v["pipeline"]["llm_agent_cjk_signal"]                 # 신호 257편 전부(라벨 증빙)
    assert cjk.cjk_signal.all() and set(cjk.id) <= set(allc.id)
    assert (cjk.lang_label != "").all(), "라벨 없는 신호 논문이 있다(D36)"
    fin = cjk[cjk.cjk_lang]                                                  # 최종 집합 = 라벨 '실제 연구 대상'
    assert len(fin) == v["pipeline"]["llm_agent_cjk_lang"]
    assert fin.lang_target_human.all() and not cjk[~cjk.cjk_lang].lang_target_human.any()
    assert allc.groupby("year").size().to_dict() == {int(k): n for k, n in v["llm_agent_by_year"].items()}
    assert fin.groupby("year").size().to_dict() == {int(k): n for k, n in v["cjk_lang_by_year"].items()}
    assert {c: int(fin[f"lang_{c}"].sum()) for c in ["ko", "zh", "ja"]} == v["cjk_lang_by_lang_total"]
    assert {c: int(cjk[f"lang_{c}_signal"].sum()) for c in ["ko", "zh", "ja"]} == v["language_label_review"]["signal_by_lang_total"]
    assert int(allc.cjk_lang.sum()) == len(fin)
    assert allc.year.between(2023, 2026).all()                                   # D30 기간
    ym = allc.year * 100 + allc.month
    assert ((ym >= 202301) & (ym <= 202609)).all(), "기간 밖 논문이 있다(2023-01~2026-09)"
    assert v["definitions"]["year_basis"] == "id"                              # D31 연도 기준
    assert (allc.year == (2000 + allc.id.str[:2].astype(int))).all()           # 연도 = id 접두
    tnorm = allc.title.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    assert not (tnorm.duplicated() & (tnorm.str.len() >= 20)).any(), "같은 제목이 남아 있다(D31 중복 제거)"
    tag = pd.read_csv(ROOT / "preprocessing/output/corpus_agent_tag_2023_2026q3.csv", dtype={"id": str})
    assert len(tag) == v["agent_tag_exploration"]["n"]
    assert set(allc.id) <= set(tag.id)          # LLM 에이전트는 agent 태그 논문의 부분집합
