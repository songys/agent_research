# 4.1 주제 변화

- 결과: 연도별 자료 규모·LDA 토픽 비중, 표 3의 연어 비율과 해석, 특징 어휘의 문서 비율.
- 정본: `overview_notebook.table1`, `topics_lda.world_topic_table`, `agent_colloc.world`, `meta_keyness.y2026_top` (`verification/*_20261015.json`).
- 코드: [`analysis/scripts/02_make_notebooks.py`](../../analysis/scripts/02_make_notebooks.py).
- 노트북: `analysis/notebooks/`의 02·07·08번.

```bash
python paper_code/run.py notebooks --notebooks 02 07 08 --output /tmp/agent-ch4_1_trends
```

전체 대응과 검증 방법은 [상위 안내](../README.md)를 따른다.
