# retrieval·graph 중국어 집합 13편의 초록 확인

입력은 고정 말뭉치에 저장된 제목·초록이다. AI 보조 검토이며 사람 판정을 대체하지 않는다. 대상 언어 라벨은 변경하지 않았다. 13편은 선택한 LDA 토픽의 배정 편수이며, 검색 기법 연구만을 별도로 판정한 편수가 아니다.

| 논문 | 초록에서 확인한 내용 |
|---|---|
| [2505.18105: ManuSearch: Democratizing Deep Search in Large Language Models with a Transparent and Open Multi-Agent Framework](https://arxiv.org/abs/2505.18105) | 세 개의 에이전트가 질의 계획·웹 검색·웹페이지에서 근거 추출을 수행한다. 영어·중국어 ORION 벤치마크를 사용한다. |
| [2506.18511: Standard Applicability Judgment and Cross-jurisdictional Reasoning: A RAG-based Framework for Medical Device Compliance](https://arxiv.org/abs/2506.18511) | 의료기기 규격을 검색하고 관할별 적용 가능성을 판정하는 RAG 시스템이다. 초록의 Chinese는 중국 규격을 가리켜 언어 대상 여부는 원문에서 추가 확인할 여지가 있다. |
| [2508.09129: BrowseMaster: Towards Scalable Web Browsing via Tool-Augmented Programmatic Agent Pair](https://arxiv.org/abs/2508.09129) | 계획기와 실행기가 웹 탐색과 근거 검색을 분담한다. 영어·중국어 벤치마크를 평가한다. |
| [2509.00366: KG-RAG: Enhancing GUI Agent Decision-Making via Knowledge Graph-Driven Retrieval-Augmented Generation](https://arxiv.org/abs/2509.00366) | 지식 그래프와 검색 증강 생성으로 GUI 에이전트의 행동 경로를 찾는다. 중국 모바일 생태계 벤치마크를 제시한다. |
| [2510.21557: Co-Sight: Enhancing LLM-Based Agents via Conflict-Aware Meta-Verification and Trustworthy Reasoning with Structured Facts](https://arxiv.org/abs/2510.21557) | 구조화한 사실과 충돌 검증으로 추론의 신뢰성을 높인다. Chinese-SimpleQA 평가를 포함하며 검색 자체보다 검증이 중심이다. |
| [2601.05520: CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems](https://arxiv.org/abs/2601.05520) | 중국 고대 역사 자료에서 사건 분류 체계를 만든다. 외부 구조화 자료를 결합하나 검색 자체가 핵심 목적은 아니다. |
| [2601.17692: LegalMALR:Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval](https://arxiv.org/abs/2601.17692) | 중국어 법률 질의를 재작성하고 법령을 검색·재순위화한다. 중국어 검색의 직접적인 비교 사례다. |
| [2605.17946: SVFSearch: A Multimodal Knowledge-Intensive Benchmark for Short-Video Frame Search in the Gaming Vertical Domain](https://arxiv.org/abs/2605.17946) | 중국 게임 분야의 짧은 동영상 장면 검색 벤치마크다. 검색 환경과 도구를 제공하고 RAG 및 계획형 에이전트를 비교한다. |
| [2606.13120: EvoBrowseComp: Benchmarking Search Agents on Evolving Knowledge](https://arxiv.org/abs/2606.13120) | 영어·중국어 웹 검색 에이전트 평가 문항을 계속 갱신하는 벤치마크다. |
| [2607.20872: LegalCiteTrust: Benchmarking Citation Trustworthiness in Chinese Long-Form Legal Research Reports](https://arxiv.org/abs/2607.20872) | 중국어 장문 법률 보고서에서 검색한 법적 근거의 인용 신뢰성을 평가한다. 검색 성능보다 검색 이후의 인용 평가에 초점이 있다. |
| [2607.27562: DeepResearch Agent System](https://arxiv.org/abs/2607.27562) | 정보 검색·웹 검색·도구 호출을 결합한 DeepResearch 시스템이다. 중국어 검색 벤치마크 결과를 보고한다. |
| [2608.10444: From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models](https://arxiv.org/abs/2608.10444) | 영어·중국어 단서로 숨은 정답을 찾는 다점 연상 추론 벤치마크다. 초록에서 검색 시스템이나 지식 그래프를 연구 대상으로 확인하지 못했다. |
| [2608.24011: SAGE: From Direct Answering to Evidence-Grounded Inference for Chinese Ancient Document Understanding](https://arxiv.org/abs/2608.24011) | 중국 고문헌 이해에서 도구로 근거를 수집하고 주장을 검증한다. 근거 수집은 포함하지만 초록의 초점은 문헌 이해다. |

한국어 연구의 비교 사례로는 중국어 법령 검색(LegalMALR), 웹 탐색(BrowseMaster), 중국 게임 분야 검색 평가(SVFSearch)를 직접 연결할 수 있다. MPAR-Bench는 검색 중심 연구로 묶기 어렵고, 나머지 일부 논문도 구조화한 지식과 근거 검증을 다룬다. 토픽 이름을 해당 토픽에 배정된 모든 논문의 개별 연구 내용과 동일시하지 않는다.

자료: `preprocessing/output/corpus_llm_agent_all_2023_2026q3.csv`, `analysis/output/tables/lda_doc_topics.csv`. 수치·검토 기록: `verification/lda_retrieval_abstract_review_20261015.json`.
