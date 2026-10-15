# agent_research

**LLM 에이전트 연구의 주제 변화와 중국어·한국어·일본어 대상 연구의 분포** 재현 코드.

arXiv 전산학 분야의 LLM 에이전트 논문 20,107편(2023.01–2026.09)을 분석한다. 대상 언어 집합은 한국어 25편·중국어 186편·일본어 17편이며, 중복을 제외하면 226편이다. 한글 및 한국어 정보처리 학술대회 논문집의 에이전트 논문 62편을 함께 비교한다.

## 설치

```bash
git clone https://github.com/songys/agent_research.git
cd agent_research
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.13.5에서 검증했다. [확인한 패키지 버전](paper_code/requirements-observed.txt)은 환경 기록이며 전체 의존성을 고정한 잠금 파일(lock file)은 아니다.

## 데이터

[v1.0.0 Release](https://github.com/songys/agent_research/releases/tag/v1.0.0)에서 `agent-research-data-20261015.tar.gz`를 내려받는다. 압축 파일에는 대용량 말뭉치와 표제어·연어 캐시가 있다.

```bash
python paper_code/data.py install /path/to/agent-research-data-20261015.tar.gz
```

arXiv 제목·초록·서지정보를 제공한다. 학술대회 자료는 제목·저자·쪽수와 문서별 판정만 제공하며, 요약·OCR 텍스트와 이를 담은 형태소 텍스트는 제외한다. 학술대회 비교는 저장된 판정의 재집계까지 재현한다.

설치 과정에서 파일 목록·크기·SHA-256을 확인한다. [입력 목록](paper_code/data_manifest.json) · [자료 설명](analysis/DATA_INVENTORY.md)

## 재현

```bash
# 저장된 문서 배정·분모·표 수치 검증
python paper_code/run.py check

# 표 1·2·3 생성 및 대조
python paper_code/run.py tables --output /tmp/agent-tables

# 자료·키워드군·연어·특징 어휘·회귀 재계산
python paper_code/run.py notebooks --output /tmp/agent-analysis

# 테스트
python -m pytest -q tests
```

`--output`은 저장소 밖의 새 폴더를 지정한다. 원본을 보존하고 사본에서 계산하며, `run.json`에 환경·입력 해시·실행 명령·정본과의 차이를 남긴다. LDA 재학습은 기본 실행에 포함되지 않는다.

## 구조

| 경로 | 내용 |
|---|---|
| `paper_code/` | 실행 도구와 원고·코드 대응표 |
| `preprocessing/` | 자료 정제 코드와 입력 |
| `analysis/` | 분석 코드·노트북·산출물 |
| `verification/` | 비교 기준이 되는 수치 정본 |
| `paper/` | 표 생성 코드·표·그림 |
| `tests/` | 입력 무결성과 실행 도구 검사 |

[상세 재현 안내](REPRODUCIBILITY.md) · [표·그림과 코드 대응](paper_code/README.md) · [검증 범위](paper_code/VALIDATION.md)
