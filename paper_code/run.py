#!/usr/bin/env python3
"""현재 원고의 저장 자료 검증·표 생성·노트북 재실행.

입력: preprocessing/output, analysis/output, verification(20261015 기준).
출력: check는 터미널만, tables/notebooks는 --output의 새 격리 사본과 실행 기록.
논문 위치: 3장 표 1, 4장 표 2·3·4/그림 1·2/본문 수치. 수집/API 호출은 하지 않는다.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "20261015"
NOTEBOOKS = {
    "01": ("01_corpus_overview", "overview_notebook", ["table1", "table0_year_basis", "same_title_dedup", "sensitivity", "table1b_lang_label_review", "table3"]),
    "02": ("02_lda", "topics_lda", ["n_docs", "vocab_size", "K", "model_selection", "topics", "world_topic_table", "project_lang", "permutation", "year_share_pct", "slope_by_seed", "stability_mean_jaccard", "lang_mention_baseline"]),
    "05": ("05_hclt", "hclt_notebook", ["n", "family_by_set", "family_lift", "year_counts"]),
    "07": ("07_agent_collocations", "agent_colloc", ["set_sizes", "world", "lang", "llm_family_union"]),
    "08": ("08_keyness", "meta_keyness", ["n_vocab", "set_sizes", "n_2026", "n_prior", "lang_top", "lang_bottom", "y2026_top", "y2026_bottom"]),
    "10": ("10_topic_covariates", "meta_topic_covariates", ["year_adjusted", "logit", "significant"]),
}
TABLES = ["tab_data_year.tex", "tab_lda_topics.tex", "tab_colloc_direction.tex", "tab_families.tex"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(expected, actual, path=""):
    """정수는 값의 일치 여부로, 실수는 저장 반올림보다 작은 1e-10의 절대 오차로 대조."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return [path + ": keys differ"]
        return [d for key in expected for d in compare(expected[key], actual[key], f"{path}.{key}")]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [path + ": lengths differ"]
        return [d for i, (a, b) in enumerate(zip(expected, actual)) for d in compare(a, b, f"{path}[{i}]")]
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        if math.isnan(expected) and math.isnan(actual):
            return []
        same = math.isclose(expected, actual, rel_tol=0, abs_tol=1e-10)
    else:
        same = expected == actual
    return [] if same else [f"{path}: {expected!r} != {actual!r}"]


def prepare(output):
    output = output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents or output in ROOT.parents:
        raise ValueError("--output은 현재 저장소 밖의 새 폴더로 지정하세요.")
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    # 링크나 하드링크를 쓰지 않아 기존 산출물이 재실행 중 바뀌지 않는다.
    paths = ["preprocessing/output", "analysis/scripts", "analysis/notebooks", "analysis/output/tables", "verification", "paper", "tests"]
    ignore = shutil.ignore_patterns("__pycache__", "*.aux", "*.log", "*.fls", "*.fdb_latexmk", "*.pdf", "hclt_ocr")
    for rel in paths:
        shutil.copytree(ROOT / rel, work / rel, ignore=ignore)
    for rel in ["analysis/output/cache/lemmas_2023_2026q3.jsonl", "analysis/output/cache/agent_colloc_2023_2026q3.jsonl", "analysis/output/hclt_tokens.csv", "analysis/output/arxiv_lemma_families_2023_2026q3.csv", "analysis/output/figures/fig2b_lda_obs_vs_exp_lang.png", "analysis/output/figures/fig2b_lda_obs_vs_exp_lang.pdf", "paper/figures/fig2b_lda_obs_vs_exp_lang.pdf", "analysis/output/figures/fig1_corpus_growth.png"]:
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    # 날짜가 바뀌어도 현재 원고의 기준 파일보다 새 파일을 자동 선택하지 않는다.
    for p in (work / "verification").glob("*.json"):
        stamp = p.stem.rsplit("_", 1)[-1]
        if stamp.isdigit() and len(stamp) == 8 and stamp > BASELINE:
            p.unlink()  # 방금 만든 격리 사본 안에서만 제외
    shutil.copytree(work / "verification", output / "expected")
    return work


def command(args, cwd, log, env=None):
    with log.open("w") as f:
        subprocess.run(args, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "tables", "notebooks"])
    parser.add_argument("--output", type=Path, help="tables/notebooks: 저장소 밖의 존재하지 않는 출력 폴더")
    parser.add_argument("--notebooks", nargs="+", choices=NOTEBOOKS, default=["01", "05", "07", "08", "10"], help="02는 LDA 재학습을 포함하며 명시적으로 선택해야 실행")
    parser.add_argument("--timeout", type=int, default=7200, help="노트북 셀당 제한 시간(초)")
    args = parser.parse_args()
    if args.action == "check":
        subprocess.run([sys.executable, str(ROOT / "analysis/scripts/17_check_paper_denominators.py")], cwd=ROOT, check=True)
        return
    if args.output is None:
        parser.error("tables/notebooks에는 --output이 필요합니다.")
    work = prepare(args.output)
    output = work.parent
    record = {"started": datetime.now().isoformat(), "action": args.action, "baseline": BASELINE,
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
              "input_hashes": {str(p.relative_to(work)): digest(p) for p in work.rglob("*") if p.is_file()},
              "steps": [], "status": "running"}
    record["runner_sha256"] = digest(Path(__file__))
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(output / "matplotlib"), IPYTHONDIR=str(output / "ipython"))
    try:
        if args.action == "tables":
            table_hashes = {name: digest(work / "paper/tables" / name) for name in TABLES}
            cmd = [sys.executable, "paper/scripts/05_make_paper_tables.py", "--lda"]
            command(cmd, work, output / "tables.log", env)
            differences = [name for name in TABLES if digest(work / "paper/tables" / name) != table_hashes[name]]
            record["steps"].append({"command": cmd, "different_tables": differences})
            if differences:
                raise ValueError(f"기존 표와 다른 출력: {differences}")
        else:
            # 같은 Python 환경의 커널을 출력 폴더 안에 등록한다(사용자 커널 설정은 보존).
            kernel = output / "jupyter/kernels/paper-reproduce"
            kernel.mkdir(parents=True)
            (kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Paper reproduction", "language": "python"}))
            env["JUPYTER_PATH"] = str(output / "jupyter") + os.pathsep + env.get("JUPYTER_PATH", "")
            for number in sorted(set(args.notebooks)):
                name, prefix, keys = NOTEBOOKS[number]
                expected = json.loads((output / "expected" / f"{prefix}_{BASELINE}.json").read_text())
                # 기존 출력이 남아 실행 성공으로 오인되지 않도록 격리 사본의 결과만 지운다.
                for p in (work / "verification").glob(f"{prefix}_2*.json"):
                    p.unlink()
                cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace", f"--ExecutePreprocessor.timeout={args.timeout}", "--ExecutePreprocessor.kernel_name=paper-reproduce", f"{name}.ipynb"]
                print("RUN", name, flush=True)
                command(cmd, work / "analysis/notebooks", output / f"{name}.log", env)
                results = sorted((work / "verification").glob(f"{prefix}_2*.json"))
                if len(results) != 1:
                    raise ValueError(f"{name}: 결과 JSON이 하나 생성되지 않았습니다.")
                actual = json.loads(results[0].read_text())
                differences = [d for key in keys for d in compare(expected[key], actual[key], key)]
                record["steps"].append({"command": cmd, "keys": keys, "result": str(results[0].relative_to(work)), "differences": differences})
                if differences:
                    raise ValueError(f"{name}: 보존한 정본과 {len(differences)}개 항목이 다릅니다. run.json을 확인하세요.")
                print("MATCH", name, flush=True)
        record["status"] = "passed"
    except Exception as e:
        record["status"] = "failed"
        record["error"] = str(e)
        raise
    finally:
        record["finished"] = datetime.now().isoformat()
        (output / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print("실행 기록:", output / "run.json", flush=True)


if __name__ == "__main__":
    main()
