"""- 격리 실행과 입력 복원의 실패 조건을 검사한다."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "paper_code" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_comparison_rejects_changed_numbers_and_missing_keys():
    run = module("run")
    assert run.compare({"n": 25, "p": .2014}, {"n": 25, "p": .2014}) == []
    assert run.compare({"n": 25}, {"n": 26})
    assert run.compare({"n": 25}, {})


def test_output_cannot_replace_repository_or_existing_directory(tmp_path):
    run = module("run")
    with pytest.raises(ValueError):
        run.prepare(ROOT / "output")
    with pytest.raises(FileExistsError):
        run.prepare(tmp_path)


def archive(path, name, body):
    with tarfile.open(path, "w:gz") as tar:
        item = tarfile.TarInfo(name)
        item.size = len(body)
        tar.addfile(item, io.BytesIO(body))


def test_restore_rejects_unexpected_member(tmp_path):
    data = module("data")
    src = tmp_path / "data.tar.gz"
    archive(src, "../outside.txt", b"a")
    with pytest.raises(ValueError):
        data.install(src, tmp_path / "repo", {"inside.txt": {"bytes": 1, "sha256": "bad"}})
    assert not (tmp_path / "outside.txt").exists()


def test_restore_checks_hash_before_writing(tmp_path):
    data = module("data")
    src = tmp_path / "data.tar.gz"
    archive(src, "inside.txt", b"a")
    with pytest.raises(ValueError, match="SHA-256"):
        data.install(src, tmp_path / "repo", {"inside.txt": {"bytes": 1, "sha256": "bad"}})
    assert not (tmp_path / "repo/inside.txt").exists()
