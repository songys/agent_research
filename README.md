# agent\_research

**LLM 에이전트 연구의 주제 변화와 중국어·한국어·일본어 대상 연구의 분포** 논문 재현 코드입니다.

2023년 1월부터 2026년 9월까지 arXiv 전산학 분야에 올라온 LLM 에이전트 논문 20,107편을 분석했습니다. 이 중 특정 언어를 다룬 논문은 한국어 25편, 중국어 186편, 일본어 17편으로, 중복을 빼면 총 226편입니다. 여기에 한글 및 한국어 정보처리 학술대회(HCLT) 논문집의 에이전트 논문 62편을 함께 비교했습니다.

## 설치

```bash
git clone https://github.com/songys/agent_research.git
cd agent_research
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.13.5 환경에서 동작을 확인했습니다. [확인한 패키지 버전](paper_code/requirements-observed.txt)은 테스트 당시의 환경 기록일 뿐, 모든 의존성을 고정하는 잠금 파일(lock file)은 아닙니다.

## 데이터

[v1.0.0 Release](https://github.com/songys/agent_research/releases/tag/v1.0.0)에서 `agent-research-data-20261015.tar.gz`를 내려받은 뒤 아래 명령어로 설치합니다. 압축 파일에는 대용량 말뭉치와 표제어·연어 캐시가 들어 있습니다.

```bash
python paper_code/data.py install /path/to/agent-research-data-20261015.tar.gz
```

* **arXiv 데이터**: 제목, 초록, 서지정보를 포함합니다.

* **학술대회 데이터**: 제목, 저자, 쪽수 및 문서별 판정 결과만 공개합니다. 저작권 문제로 요약문, OCR 원문, 형태소 분석 텍스트는 제외했으므로, 학술대회 비교는 저장된 판정 결과를 다시 집계하는 단계까지만 재현할 수 있습니다.

설치 스크립트를 실행하면 파일 목록과 크기, SHA-256 해시값을 자동으로 검증합니다. ([입력 목록](paper_code/data_manifest.json) · [자료 설명](analysis/DATA_INVENTORY.md))

## 재현

```bash
# 저장된 문서 배정·분모·표 수치 검증
python paper_code/run.py check

# 표 1·2·3·4 생성 및 대조
python paper_code/run.py tables --output /tmp/agent-tables

# 자료·키워드군·연어·특징 어휘·회귀 재계산
python paper_code/run.py notebooks --output /tmp/agent-analysis

# 테스트
python -m pytest -q tests
```

`--output`에는 저장소 외부의 새 경로를 지정해야 합니다. 원본 데이터를 건드리지 않고 사본에서 계산을 수행하며, 실행 환경과 입력 해시, 실행 명령어, 정본과의 대조 결과는 `run.json`에 기록됩니다. 기본 실행 과정에는 LDA 모델 재학습이 포함되어 있지 않습니다.

## 구조

| 경로               | 설명                       |
| ---------------- | ------------------------ |
| `paper_code/`    | 실행 스크립트 및 논문·코드 대응표      |
| `preprocessing/` | 데이터 정제 코드와 원본 입력         |
| `analysis/`      | 분석 스크립트·노트북·산출물          |
| `verification/`  | 결과 대조용 기준 수치(정본)         |
| `paper/`         | 논문에 수록된 표 생성 코드·표·그림     |
| `tests/`         | 입력 데이터 무결성 및 실행 스크립트 테스트 |

[상세 재현 안내](REPRODUCIBILITY.md) · [표·그림과 코드 대응](paper_code/README.md) · [검증 범위](paper_code/VALIDATION.md)

## 인용

이 코드나 데이터를 사용하실 때는 아래 논문을 인용해 주세요.

> Youngsook Song. 2026. 에이전트 연구의 주제 변화와 한중일 언어권별 분포 [Topic shifts in agent research and its distribution across Korean, Chinese, and Japanese language communities]. *Communications of the Korean Institute of Information Scientists and Engineers (정보과학회지)*, 44(11), 46–51.

```bibtex
@article{song2026agent,
  author  = {Song, Youngsook},
  title   = {에이전트 연구의 주제 변화와 한중일 언어권별 분포 [Topic shifts in agent research and its distribution across Korean, Chinese, and Japanese language communities]},
  journal = {Communications of the Korean Institute of Information Scientists and Engineers (정보과학회지)},
  volume  = {44},
  number  = {11},
  pages   = {46--51},
  year    = {2026}
}
```
