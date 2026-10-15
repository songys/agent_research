# 자료 목록

| 자료 | 경로 | 규모·역할 | 배포 |
|---|---|---|---|
| arXiv 원자료 | 원자료 수집본 | 다섯 분류의 수집본 | 원자료 별도 |
| 선정 말뭉치 | `preprocessing/output/corpus_llm_agent_all_2023_2026q3.csv` | 20,107편 | Release |
| 언어 신호와 판정 | `preprocessing/output/corpus_llm_agent_cjk_lang_2023_2026q3.csv` | 신호 257편, 대상 언어 합집합 226편 | Git |
| agent 태그 말뭉치 | `preprocessing/output/corpus_agent_tag_2023_2026q3.csv` | 선정 규칙 밖 논문의 보조 집계 | Release |
| 학술대회 메타데이터 | `preprocessing/output/hclt_papers.csv` | 제목·저자·쪽수·판정: 525편 중 에이전트 62편 | Git |
| 사람 판정 | `verification/review_*.jsonl` | 언어·학술대회 판정 | Git |
| 표제어·인접 연어 | `analysis/output/cache/{lemmas,agent_colloc}_2023_2026q3.jsonl` | 제목·초록에서 추출한 중간 입력 | Release |
| 키워드군 판정 | `analysis/output/{hclt_tokens,arxiv_lemma_families_2023_2026q3}.csv` | 문서별 키워드군 판정 | Git |
| 토픽 배정 | `analysis/output/tables/lda_doc_topics.csv` | 문서별 LDA 토픽 | Git |
| 정본 | `verification/*_20261015.json` | 원고 수치 대조 | Git |

말뭉치 해시는 `preprocessing/output/corpus_manifest.json`, 배포 입력 해시는 `paper_code/data_manifest.json`에 있다. 언어별 25·186·17편의 합은 중복을 포함한 228편이다. arXiv 자료는 메타데이터이고, 학술대회 원문 PDF와 OCR 원문은 Release에 포함하지 않는다.

학술대회 요약·OCR 텍스트·주제어 텍스트·명사와 표제어 열은 Git과 Release에 포함하지 않는다. `hclt_tokens.csv`는 파일명만 유지하며 문서 식별자와 판정 열만 담는다. 학술대회 자료 출처: [논문집 안내](http://www.hclt.kr/symp/?lnb=conference).

## 추가 검증

12개 LDA 설정의 배정·단어는 `analysis/output/tables/lda_sensitivity_runs_20261015.json`, 비교 수치는 `verification/lda_sensitivity_20261015.json`에 있다. `analysis/notebooks/13_lda_sensitivity.ipynb`에서 재집계한다. OCR 검증 결과는 `verification/hclt_ocr_audit_20261015.json`이며, 원문과 보정 텍스트는 배포하지 않는다. 현재 본문은 회귀·순열 검정값 대신 설정 변화와 개별 연구 사례를 설명한다.
