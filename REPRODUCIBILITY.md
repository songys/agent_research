# 재현 안내

수치 정본의 기준일은 2026-10-15이다. 설치와 기본 실행은 [README](README.md)를 따른다.

## 실행 범위

| 명령 | 계산 | 대조 대상 |
|---|---|---|
| `run.py check` | 저장된 문서 배정·언어 판정 재집계 | 분모·관측·기대 편수·표 2 |
| `run.py tables` | 정본에서 LaTeX 표 생성 | 표 1·2·3의 바이트 일치 |
| `run.py notebooks` | 01·05·07·08·10번 전체 실행 | 원고 관련 JSON 키 |
| `run.py notebooks --notebooks 01 02 05 07 08 10` | 위 집계와 LDA 재학습·순열 검정 | 토픽 배정·추세·검정값 포함 |

노트북은 실행 명령과 동일한 Python으로 시작한다. 실행 전 보존한 정본과 비교하며, 정수는 정확히, 실수는 절대 오차 1e-10 이내로 대조한다. 차이가 있으면 실패로 기록하고 원본은 갱신하지 않는다. 비교하는 JSON 키는 `paper_code/run.py`의 `NOTEBOOKS`에 있다.

## LDA 재학습

```bash
python paper_code/run.py notebooks --notebooks 01 02 05 07 08 10 --output /tmp/agent-full
```

배포한 표제어 캐시를 사용한다. 02번은 토픽 수 후보 5개와 난수 시드(random seed) 42·43·44의 학습을 포함한다. 시드별 토픽은 상위 단어의 자카드 유사도(Jaccard similarity)로 대응한다. 패키지·실행 환경에 따른 수치 차이는 대조 결과에 남긴다.

캐시를 다시 추출하려면 `en_core_web_sm 3.8.0`과 Kiwi가 필요하다. 연어 추출은 `analysis/scripts/12_agent_collocations.py`, 키워드군 추출은 `09_hclt_morph.py`에 있다. 기본 재현은 배포된 캐시와 키워드군 판정을 사용한다.

## 입력과 출력

- 대용량 입력 4개는 [Release](https://github.com/songys/agent_research/releases/tag/v1.0.0)로 제공한다. `paper_code/data.py install`이 SHA-256을 대조한다.
- 자료의 구성은 [자료 목록](analysis/DATA_INVENTORY.md), 결과 대응은 [코드 안내](paper_code/README.md)에 있다.
- 원자료 PDF와 전체 수집본은 포함하지 않는다. arXiv는 정제 말뭉치부터, 학술대회는 저장된 문서별 판정부터 재집계한다.
- 수집 API·OpenAlex 조회·BERTopic·STM 학습은 실행하지 않는다. 10번은 저장된 STM을 보조 비교에 읽으며, 원고 수치 대조에는 회귀·연도 조정 결과만 사용한다.
- 그림은 기본 스타일로 생성한다. 수치 대조와 이미지의 바이트 일치는 구분한다.

확인한 범위와 미실행 항목은 [검증 기록](paper_code/VALIDATION.md)에 있다.

학술대회 요약·OCR 텍스트 및 형태소 텍스트는 배포하지 않는다. 학술대회 키워드 판정을 처음부터 다시 추출하려면 원문 요약을 별도로 확보하고 경계를 검수해야 한다. 공개 판정은 기존 분석에 사용한 값이며, OCR 요약 경계 오류를 교정한 재분석 결과는 아니다. 상위 명사 탐색은 공개 재현에서 제외한다.

## 원고 그림

```bash
python analysis/scripts/19_plot_paper_figures.py
```

저장된 정본에서 그림 1(연도별 편수·비율)과 그림 2(대상 언어별 토픽 분포)를 만든다. 토픽명은 영어 원문을 유지하고 설명은 국문으로 표시한다. NanumGothic, Noto Sans CJK KR, AppleGothic, Malgun Gothic 중 하나가 필요하다. 분석과 모델 학습은 다시 실행하지 않는다.

표 2에는 기울기를 표시하지 않는다. 기존 정본의 기울기와 반복 학습 값은 분석 기록으로 보존하며, 원고의 주제 변화는 연도별 비중으로 설명한다. 토픽별 검정은 선택한 전처리와 LDA 모델에 조건부인 결과이다. 전처리·토픽 수 변경에 따른 체계적인 민감도 분석(sensitivity analysis)은 완료하지 않았다.

원고의 ‘전체 집합’은 선정한 arXiv LLM 에이전트 논문 20,107편이며, 한중일어 대상 연구를 포함한다. 정본 JSON의 기존 `world` 키는 이 집합을 가리킨다.
