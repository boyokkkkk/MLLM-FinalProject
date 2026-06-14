from src.models.retrieval import Evidence
from src.serving.api import (
    _append_inline_citation_summary,
    _build_citations,
    _looks_truncated_answer,
    _normalize_visual_assist_policy,
    _postprocess_answer,
    _should_enable_visual_assist,
)


def test_postprocess_answer_preserves_markdown_structure() -> None:
    raw = """Answer: ## Summary

- **Alpha**
- `beta`

| col | value |
| --- | --- |
| a | 1 |
"""

    cleaned = _postprocess_answer(raw)

    assert cleaned.startswith("## Summary")
    assert "- **Alpha**" in cleaned
    assert "- `beta`" in cleaned
    assert "| col | value |" in cleaned


def test_truncated_answer_detects_unclosed_math_and_parentheses() -> None:
    broken = "Step 3: solve for $\\mathbf{w}^*$ with gradient $\\nabla_{\\mathbf{w}} L(\\mathbf{w}"
    assert _looks_truncated_answer(broken)


def test_truncated_answer_ignores_complete_short_answer() -> None:
    complete = "The final result is $w^*=(X^TX)^{-1}X^Ty$."
    assert not _looks_truncated_answer(complete)


def test_build_citations_extracts_explicit_locator_fields() -> None:
    evidence = Evidence(
        chunk_id="chk-1",
        source="docvqa/val/10269#page=3#figure=fig-12#block=block-0002",
        page=3,
        text="Example block",
        snippet="Example block",
        score=1.0,
    )

    citation = _build_citations([evidence])[0]

    assert citation.page == 3
    assert citation.figure_id == "fig-12"
    assert citation.figure_no == "12"
    assert citation.block_id == "block-0002"


def test_append_inline_citation_summary_adds_page_and_figure_labels() -> None:
    evidence = Evidence(
        chunk_id="chk-1",
        source="docvqa/val/10269#page=3#figure=fig-12#block=block-0002",
        page=3,
        text="Example block",
        snippet="Example block",
        score=1.0,
    )
    citation = _build_citations([evidence])[0]

    answer = _append_inline_citation_summary("The answer is 42.", [citation])

    assert "Citations:" in answer
    assert "[1] [p.3, fig-12]" in answer


def test_visual_assist_policy_normalization_and_gate_behavior() -> None:
    evidence = Evidence(
        chunk_id="chk-1",
        source="docvqa/val/10269#page=3#figure=fig-12#block=block-0002",
        page=3,
        text="Example block",
        snippet="Example block",
        score=1.0,
        chunk_type="figure",
    )

    assert _normalize_visual_assist_policy("gated") == "gated_logo_pack_heading_title_page_handwritten_only"
    assert _normalize_visual_assist_policy("always") == "always"
    assert _normalize_visual_assist_policy("off") == "off"
    assert _should_enable_visual_assist("What is the page number?", [evidence], "always")
    assert not _should_enable_visual_assist("What is the page number?", [evidence], "off")
