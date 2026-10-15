#!/usr/bin/env python3
"""- pack: 재현 입력 4개를 SHA-256 확인 후 압축한다.
- install: 압축 파일의 경로·해시를 확인하여 저장소에 복원한다.
- 입력 목록: data_manifest.json. 원자료 PDF와 수집본은 포함하지 않는다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("data_manifest.json")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def install(archive, root, files):
    with tarfile.open(archive, "r:gz") as tar, tempfile.TemporaryDirectory() as tmp:
        members = tar.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError("압축 파일의 입력 목록이 data_manifest.json과 다릅니다.")
        for member in members:
            info = files[member.name]
            if not member.isfile() or member.size != info["bytes"]:
                raise ValueError("일반 파일이 아니거나 크기가 다릅니다: " + member.name)
            dest = (root / member.name).resolve()
            if root.resolve() not in dest.parents:
                raise ValueError("저장소 밖 경로입니다: " + member.name)
            if dest.exists() and sha(dest) != info["sha256"]:
                raise ValueError("기존 파일이 다릅니다. 덮어쓰지 않습니다: " + member.name)
            staged = Path(tmp) / member.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, staged.open("wb") as out:
                import shutil
                shutil.copyfileobj(src, out)
            if sha(staged) != info["sha256"]:
                raise ValueError("SHA-256 불일치: " + member.name)
        # 모든 파일을 검증한 뒤에만 반영한다.
        for name in files:
            dest = root / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(Path(tmp) / name, dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["pack", "install"])
    p.add_argument("archive", type=Path)
    args = p.parse_args()
    files = json.loads(MANIFEST.read_text())["files"]
    if args.action == "pack":
        for name, info in files.items():
            if sha(ROOT / name) != info["sha256"]:
                raise ValueError("입력 해시 불일치: " + name)
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with args.archive.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz") as tar:
            for name in files:
                tar.add(ROOT / name, arcname=name, recursive=False)
        args.archive.with_suffix(args.archive.suffix + ".sha256").write_text(sha(args.archive) + "  " + args.archive.name + "\n")
    else:
        install(args.archive, ROOT, files)
    print(args.action, args.archive)


if __name__ == "__main__":
    main()
