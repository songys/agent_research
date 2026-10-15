"""입력: LDA 민감도 정본 JSON. 출력: 결과 안내서. 논문 위치: 4.1·4.2."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
j=json.loads((ROOT/'verification/lda_sensitivity_20261015.json').read_text())
runs=j['runs']
lines=['# LDA 설정 민감도 분석 결과 — 2026-10-15','',
'전체 20,107편과 한국어 25편·중국어 186편·일본어 17편을 유지한 채 토픽 수(K) 10/25, 현행 불용어 및 datum·cod를 남긴 불용어, 난수 시드(random seed) 42/43/44를 조합한 12개 모델을 학습했다. 기존 K25·seed42의 문서 배정과 상위 단어가 재현되었다.','',
'현재 원고의 학습·미세조정 집중과 실행·검색 0편은 모델 설정의 영향을 받았다. 소프트웨어 실행에 대응시킨 토픽의 2026년 비중은 2023년보다 높았으며, 대응 토픽을 구성하는 단어와 논문은 설정에 따라 달라졌다.','',
'## 계산과 대응 방법','',
'원고의 training·fine(T6), software·execution(T20), retrieval·graph(T22)를 기준으로 새 모델의 상위 10개 단어가 가장 많이 겹치는 토픽과 배정 논문이 가장 많이 겹치는 토픽을 각각 찾았다. Jaccard 유사도는 교집합 크기를 합집합 크기로 나눈 값이다. 서로 다른 두 대응을 합쳐 하나의 일치 판정으로 처리하지 않았다. K10에서는 토픽의 병합·분할이 포함될 수 있다. 연도별 비중의 분모는 해당 연도의 전체 집합이며 2026년은 1~9월이다.','',
'## 현행 불용어·K25의 세 개 시드','',
'| 시드 | 기준 토픽 | 대응 토픽 상위 두 개 단어 | 한국어 편수 | 중국어 편수 | 단어 Jaccard | 문서 Jaccard |','|---|---|---|---:|---:|---:|---:|']
for r in runs:
 if r['mode']=='current' and r['k']==25:
  for name,t in r['targets'].items():
   v=t['matches']['top_words'];lines.append(f"| {r['seed']} | {name} | {'·'.join(v['top_words'][:2])} | {v['language_n']['ko']} | {v['language_n']['zh']} | {v['word_jaccard']:.2f} | {v['document_jaccard']:.2f} |")
lines+=['',
'한국어 training·fine에 배정된 편수는 다른 시드의 대응 토픽에서 유지되지 않았다. 낮은 문서 유사도는 기존 논문들이 다른 토픽으로 분산되었음을 뜻한다. 특정 모델에서 6편이 가장 큰 토픽을 이루었다는 관측은 남길 수 있으며, 한국어 연구 전반이 학습·미세조정에 집중되어 있다는 일반화에는 추가 근거가 필요하다.','',
'## 모든 설정의 비교','',
'다음 표는 상위 단어를 기준으로 대응한 결과다. 단어 대응과 문서 대응이 다르면 마지막 열에 표시했다. 한국어 비율비는 한국어 집합의 대응 토픽 비중을 전체 집합의 대응 토픽 비중으로 나눈 값이며, 연도·분류를 보정한 오즈비(odds ratio)와 다르다.','',
'| 설정 | 기준 토픽 | 한국어 편수 | 중국어 편수 | 한국어 비율비 | 단어 Jaccard | 문서 Jaccard | 2026−2023 비중 차이(pp) | 대응 일치 |','|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in runs:
 for name,t in r['targets'].items():
  v=t['matches']['top_words'];lines.append(f"| {r['name']} | {name} | {v['language_n']['ko']} | {v['language_n']['zh']} | {v['ko_share_ratio_to_overall']:.2f} | {v['word_jaccard']:.2f} | {v['document_jaccard']:.2f} | {v['change_2026_minus_2023_pp']:.2f} | {'같음' if t['matching_methods_agree'] else '다름'} |")
for name in ['training','software','retrieval']:
 vals=[r['targets'][name]['matches']['top_words']['language_n']['ko'] for r in runs]
 lines+=['',f"- {name}의 단어 대응 토픽에 배정된 한국어 편수 범위: {min(vals)}~{max(vals)}편."]
for method,label in [('top_words','단어'),('documents','문서')]:
 vals=[r['targets']['software']['matches'][method]['change_2026_minus_2023_pp'] for r in runs]
 lines += [f"- {label} 대응 software 토픽의 2026−2023 비중 차이: {min(vals):.2f}~{max(vals):.2f}pp. 양수인 설정 {sum(v>0 for v in vals)}/{len(vals)}개."]
lines+=['',
'단어 대응의 software 편수는 0편에서 벗어나는 설정이 있었다. retrieval은 K25에서 datum·cod를 남긴 seed42 모델에서도 한국어 1편이 배정되었다. 이 대응의 단어 Jaccard는 0.82이므로 낮은 단어 유사도만으로 해당 변화를 설명할 수 없다. K10에서 한국어 11편이 배정된 대응 토픽은 추론과 검색을 함께 포함한다. 토픽 병합의 영향을 받는 11편을 검색 연구만의 증가로 읽지 않는다.','',
'소프트웨어 실행에 관한 방향은 두 대응 방식 모두에서 양수였다. 모든 설정에서 같은 토픽 경계를 재현한 결과는 아니다. 현재 모델의 비중 변화와 인접 연어·특징 어휘의 관측을 함께 제시하고, 설정별 범위를 동등한 토픽의 추정 구간으로 쓰지 않는다.','',
'## 중국어 retrieval·graph 13편','',
'13편은 기준 모델에서 배정된 논문 수다. 다른 모델의 단어 대응 토픽에 남는 문서와 새로 들어오는 문서가 다르다. 제목·초록의 AI 보조 검토에서 법령 검색, 웹 탐색, 검색 평가의 직접적인 사례와 분류 체계 구성·근거 검증·연상 추론의 사례가 함께 확인되었다. `LDA_RETRIEVAL_ABSTRACT_REVIEW_20261015.md`에서 논문별 근거를 확인할 수 있다. 사람 검수나 대상 언어 라벨 변경은 수행하지 않았다.','',
'## 원고에 반영할 범위','',
'- 표 2의 수치는 기존 모델의 관측치로 유지한다. 설정에 따른 문서 재배정이 확인되었음을 본문에 추가한다.','- 한국어 학습·미세조정 6편과 실행·검색 0편을 한국어 연구 전반의 고정된 특성으로 일반화하지 않는다.','- 실행·검색의 검토 후보 제시는 전체 집합의 관측 변화, 키워드군별 관측, 중국어 관련 논문의 구체적인 사례에 연결한다.','- 기존 보정 오즈비와 p값은 기준 모델의 배정에서 얻은 값이다. 설정별 회귀·순열 검정은 이번 범위에 포함하지 않았으며, 원고에서는 이 검정 결과의 서술을 덜고 설정에 따른 변화와 사례를 제시하였다.','',
'## 재현','',
'```bash\npython analysis/scripts/20_lda_sensitivity.py\npython analysis/scripts/21_report_lda_sensitivity.py\n```','',
'- 학습 출력: `analysis/output/tables/lda_sensitivity_runs_20261015.json`','- 수치 정본: `verification/lda_sensitivity_20261015.json`','- 빈 커널 재집계: `analysis/notebooks/13_lda_sensitivity.ipynb`','- 실행 기록: `verification/lda_sensitivity_notebook_run_20261015.json`','- 계획: `notes/LDA_SENSITIVITY_PLAN_20261015.md`','',
'노트북은 저장된 모델 배정에서 비교 수치를 다시 계산하며, 모델 학습을 매번 반복하지 않는다. 별도 pandas 교차표로 편수와 연도별 비중을 대조했다. 시험한 설정은 가능한 선택의 일부이며, 한국어 집합의 작은 크기는 남는다.']
(ROOT/'notes/LDA_SENSITIVITY_WALKTHROUGH_20261015.md').write_text('\n'.join(lines)+'\n')
