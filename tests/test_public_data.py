"""학술대회 공개 자료의 열과 Release 입력 범위 검사."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_conference_public_columns():
    allowed = {
        "preprocessing/output/hclt_papers.csv": {
            "id", "year", "printed_page", "pdf_page", "title_ko", "authors_ko",
            "source", "agent_tag", "llm_tag", "is_llm_agent", "is_llm_agent_human", "kw_families",
        },
        "analysis/output/hclt_tokens.csv": {
            "id", "year", "source", "is_llm_agent_morph", "is_llm_agent_human",
            *["f_" + name for name in (
                "agent", "llm", "rag", "transformer", "multimodal", "diffusion",
                "reinforcement_learning", "robotics", "fine_tuning", "tabular",
                "question_answering", "translation", "tool", "simulation", "human",
                "safety", "evaluation", "physical",
            )],
        },
    }
    for rel, columns in allowed.items():
        with (ROOT / rel).open(newline="") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == columns
            assert len(list(reader)) == 525


def test_release_excludes_conference_text():
    manifest = json.loads((ROOT / "paper_code/data_manifest.json").read_text())
    assert "hclt" not in json.dumps(manifest).lower()
    for path in (ROOT / "analysis/notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        assert all(not c.get("outputs") for c in notebook["cells"])
