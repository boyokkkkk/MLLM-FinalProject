from src.models.retrieval import Evidence
from src.serving.api import (
    _build_retrieval_query,
    _build_uploaded_image_evidences,
    _is_exhaustive_question_request,
    _is_followup_query,
    _merge_evidences,
    _select_citation_evidences,
    _should_prefer_uploaded_images_only,
    _should_prefer_workspace_only,
)


def _evidence(chunk_id: str, *, kind: str, source: str, text: str, section_title: str | None = None) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        source=source,
        page=1,
        text=text,
        snippet=text,
        score=1.0,
        section_title=section_title,
        citation_kind=kind,
    )


def test_exhaustive_question_request_detection() -> None:
    assert _is_exhaustive_question_request("回答图中的所有题目")
    assert _is_exhaustive_question_request("answer all questions in the image")
    assert not _is_exhaustive_question_request("summarize the report")


def test_followup_query_expands_with_recent_user_history() -> None:
    history = [
        {"role": "user", "content": "回答图中的所有题目"},
        {"role": "assistant", "content": "已回答第一题。"},
    ]
    assert _is_followup_query("第二题呢")
    expanded = _build_retrieval_query("第二题呢", history)
    assert "回答图中的所有题目" in expanded
    assert "第二题呢" in expanded


def test_workspace_first_can_suppress_corpus_for_visual_workspace_question() -> None:
    workspace = [_evidence("w1", kind="workspace_indexed", source="workspace_file:test.jpg", text="题目一")]
    corpus = [_evidence("c1", kind="corpus", source="docvqa/val/1", text="irrelevant")]
    merged = _merge_evidences(
        workspace_evidences=workspace,
        retrieved_evidences=corpus,
        request_context_evidences=[],
        scope="workspace-first",
        prefer_workspace_only=True,
    )
    assert [item.chunk_id for item in merged] == ["w1"]
    assert _should_prefer_workspace_only("workspace-first", "回答图中的所有题目", [], workspace)


def test_exhaustive_workspace_query_returns_more_workspace_citations() -> None:
    evidences = [
        _evidence("w1", kind="workspace_indexed", source="workspace_file:test.jpg", text="题目一", section_title="题目一"),
        _evidence("w2", kind="workspace_indexed", source="workspace_file:test.jpg", text="题目二", section_title="题目二"),
        _evidence("c1", kind="corpus", source="docvqa/val/1", text="irrelevant"),
    ]
    selected = _select_citation_evidences(evidences, "workspace-first", "回答图中的所有题目")
    assert [item.chunk_id for item in selected] == ["w1", "w2"]


def test_uploaded_temporary_images_can_become_local_evidence() -> None:
    evidences = _build_uploaded_image_evidences(["data:image/png;base64,abc"])
    assert len(evidences) == 1
    assert evidences[0].source == "temporary_image:1"
    assert evidences[0].citation_kind == "workspace"
    assert evidences[0].inline_image_data_url == "data:image/png;base64,abc"


def test_uploaded_temporary_images_should_not_fall_back_to_global_corpus() -> None:
    assert _should_prefer_uploaded_images_only(
        query="回答图中的所有题目",
        image_data_urls=["data:image/png;base64,abc"],
        workspace_evidences=[],
    )
