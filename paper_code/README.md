# 원고와 코드 대응

대상 논문: LLM 에이전트 연구의 주제 변화와 중국어·한국어·일본어 대상 연구의 분포. 아래 정본 파일은 별도 표기가 없으면 `verification/<이름>_20261015.json`이다. 검수 칸은 저자가 재실행과 원고 대조 후 표시한다.

| 원고 | 결과 | 정본 이름과 키 | 코드·노트북 | 검수 |
|---|---|---|---|---|
| 전체 | 분모·관측·기대 편수 | `corpus_counts.pipeline`, `topics_lda.world_topic_table` | [분모 검증](../analysis/scripts/17_check_paper_denominators.py) | [ ] |
| 3장 표 1 | 연도별 자료 규모 | `overview_notebook.table1` | [01](../analysis/notebooks/01_corpus_overview.ipynb) | [ ] |
| 4장 표 2 | 25개 토픽과 언어 분포 | `topics_lda.world_topic_table`, `lang_mention_baseline` | [02](../analysis/notebooks/02_lda.ipynb) | [ ] |
| 4장 그림 1 | 두 기준의 기대 편수 | `topics_lda.project_lang` | [그림 생성](../analysis/scripts/19_plot_paper_figures.py) | [ ] |
| 4.1 그림 2 | 연도별 편수·비율 | `overview_notebook.table1` | [그림 생성](../analysis/scripts/19_plot_paper_figures.py) | [ ] |
| 4장·4.1 | 토픽 구성의 안정성과 연도별 비중 | `topics_lda.stability_mean_jaccard`, `world_topic_table` | [02](../analysis/notebooks/02_lda.ipynb) | [ ] |
| 4.1.1 | 인접 연어의 비율 | `agent_colloc.world`, `set_sizes` | [추출](../analysis/scripts/12_agent_collocations.py), [07](../analysis/notebooks/07_agent_collocations.ipynb) | [ ] |
| 4.1.1 | 특징 어휘의 비율 | `meta_keyness.y2026_top`, `n_2026`, `n_prior` | [08](../analysis/notebooks/08_keyness.ipynb) | [ ] |
| 4.2.1 | 전체 분포 순열 검정 | `topics_lda.permutation` | [02](../analysis/notebooks/02_lda.ipynb) | [ ] |
| 4.2.1 | 연도 조정 회귀 | `meta_topic_covariates.logit`, `year_adjusted` | [10](../analysis/notebooks/10_topic_covariates.ipynb) | [ ] |
| 4.2.2 표 3 | 키워드군 비율 | `hclt_notebook.family_by_set` | [형태소 처리](../analysis/scripts/09_hclt_morph.py), [05](../analysis/notebooks/05_hclt.ipynb) | [ ] |
| 표 1·2·3 | LaTeX 표 | 위 세 정본 | [표 생성](../paper/scripts/05_make_paper_tables.py) `--lda` | [ ] |

## 실행

```bash
python paper_code/run.py check
python analysis/scripts/19_plot_paper_figures.py
python paper_code/run.py tables --output /tmp/agent-tables
python paper_code/run.py notebooks --output /tmp/agent-analysis
```

[설치·데이터](../README.md) · [LDA 재학습](../REPRODUCIBILITY.md) · [검증 기록](VALIDATION.md)

## 장별 안내

- [3장 자료](ch3_data/README.md)
- [4.1 주제 변화](ch4_1_trends/README.md)
- [4.2 언어별 분포와 비교](ch4_2_languages/README.md)

기존 스크립트를 이동하거나 복제하지 않고 `run.py`에서 연결한다.
