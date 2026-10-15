# 결과와 코드 대응

현재 원고의 대응표는 [paper_code/README.md](../paper_code/README.md)에 통합하였다. 각 행은 원고 위치, JSON 키, 실행 코드, 노트북을 연결한다.

## 추가 검증

12개 LDA 설정의 배정·단어는 `analysis/output/tables/lda_sensitivity_runs_20261015.json`, 비교 수치는 `verification/lda_sensitivity_20261015.json`에 있다. `analysis/notebooks/13_lda_sensitivity.ipynb`에서 재집계한다. OCR 검증 결과는 `verification/hclt_ocr_audit_20261015.json`이며, 원문과 보정 텍스트는 배포하지 않는다. 현재 본문은 회귀·순열 검정값 대신 설정 변화와 개별 연구 사례를 설명한다.
