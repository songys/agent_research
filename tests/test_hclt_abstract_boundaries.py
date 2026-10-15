"""요약 경계를 찾지 못한 OCR 텍스트가 요약으로 대체되지 않는지 검사."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "preprocessing/scripts/07_hclt_corpus.py"
spec = importlib.util.spec_from_file_location("hclt_corpus", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_boundaries_do_not_create_abstract():
    for text in (
        "제목\n저자\n1. 서론\n본문 내용",
        "제목\n요약\n요약인지 확인하지 못한 내용",
        "제목\n본문 내용\n주제어: 에이전트",
    ):
        assert module.parse_first_page(module.normalize_ocr(text))["abstract_ko"] == ""


def test_explicit_boundaries_exclude_body():
    text = "제목\n요약\n초록 내용\n주 제 어: 에이전트\n1. 서론\n본문 내용"
    assert module.parse_first_page(module.normalize_ocr(text))["abstract_ko"] == "초록 내용"


def test_abstract_without_keywords_ends_before_introduction():
    text = '제목\n요약\n초록 내용\n1.\n서론\n본문의 에이전트 용어'
    assert module.parse_first_page(text)['abstract_ko'] == '초록 내용'


def test_zero_width_markers_and_english_keywords():
    text = '제목\n\u200b요 약\u200b\n초록 내용\n\u200b주제어: 평가\n1. 서론\n본문'
    result = module.parse_first_page(text)
    assert result['abstract_ko'] == '초록 내용'
    assert result['keywords_ko'] == '평가'
    text = '제목\nAbstract\n초록 내용\nSubject word: speech recognition\n1. 서론\n본문'
    assert module.parse_first_page(text)['abstract_ko'] == '초록 내용'
