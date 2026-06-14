from src.serving.workspaces import _extract_focus_phrases, _query_match_bonus


def test_extract_focus_phrases_prefers_tail_concept_for_definition_query() -> None:
    phrases = _extract_focus_phrases("解释chapter1课件中计算机网络中网络边缘是什么？")
    assert "网络边缘" in phrases


def test_query_match_bonus_prefers_explanatory_body_over_title_only() -> None:
    query = "解释chapter1课件中计算机网络中网络边缘是什么？"
    body_bonus = _query_match_bonus(query, "网络边缘: 主机、接入网络、物理媒体", None)
    title_bonus = _query_match_bonus(query, "网络边缘和网络核心", None)
    assert body_bonus > title_bonus
