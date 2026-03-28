"""backend.services.chat.request_builder の active_drifts 対応テスト。

test_system_prompt.py と同じスタイルで、SELF_DRIFT ブロックの
挿入・非挿入・内容・位置を検証する独立ファイル。
"""

from backend.services.chat.request_builder import build_system_prompt, CHOTGOR_SELF_DRIFT_GUIDE


def test_no_active_drifts_does_not_include_self_drift_block():
    """active_drifts を渡さない場合、SELF_DRIFT ブロックが含まれないこと。"""
    prompt = build_system_prompt("You are a cat.")

    assert "## 現在有効なSELF_DRIFT" not in prompt


def test_active_drifts_includes_self_drift_block():
    """active_drifts=[...] を渡すと「## 現在有効なSELF_DRIFT」ブロックが含まれること。"""
    prompt = build_system_prompt(
        "You are a cat.",
        active_drifts=["もっとクールに話す"],
    )

    assert "## 現在有効なSELF_DRIFT" in prompt


def test_active_drifts_content_appears_in_prompt():
    """active_drifts の各内容テキストがプロンプトに含まれること。"""
    drifts = ["指針その一", "指針その二", "指針その三"]
    prompt = build_system_prompt(
        "You are a cat.",
        active_drifts=drifts,
    )

    for content in drifts:
        assert content in prompt


def test_active_drifts_block_position():
    """active_drifts ブロックが Provider ブロックの後かつ Chotgor ブロックの前にあること。

    ブロック順序:
        ... → (Provider Instructions) → (inner_narrative) → SELF_DRIFT → Chotgor 記憶ブロック
    provider_additional_instructions を指定した場合の位置を検証する。
    """
    prompt = build_system_prompt(
        "You are a cat.",
        provider_additional_instructions="Use formal language.",
        active_drifts=["クールな話し方"],
    )

    # Provider ブロックより後ろに SELF_DRIFT ブロックがあること
    provider_pos = prompt.index("## Provider-specific Instructions")
    drift_pos = prompt.index("## 現在有効なSELF_DRIFT")
    chotgor_pos = prompt.index("## あなたの記憶について")

    assert provider_pos < drift_pos < chotgor_pos


def test_chotgor_self_drift_guide_always_included():
    """CHOTGOR_SELF_DRIFT_GUIDE の説明テキスト（[DRIFT:...] の使い方）が常に含まれること。

    active_drifts の有無に関わらず、Chotgor ブロック末尾に DRIFT の使い方が
    常に記載されていることを検証する。
    """
    prompt_without_drifts = build_system_prompt("You are a cat.")
    prompt_with_drifts = build_system_prompt(
        "You are a cat.",
        active_drifts=["指針A"],
    )

    # [DRIFT:...] の使い方説明が両方に含まれること
    assert "[DRIFT:" in prompt_without_drifts
    assert "SELF_DRIFT" in prompt_without_drifts

    assert "[DRIFT:" in prompt_with_drifts
    assert "SELF_DRIFT" in prompt_with_drifts


def test_empty_active_drifts_does_not_include_self_drift_block():
    """active_drifts=[]（空リスト）の場合に SELF_DRIFT ブロックが含まれないこと。

    空リストは「現在有効な指針がない」状態であり、ブロック自体を挿入しないことを検証する。
    """
    prompt = build_system_prompt(
        "You are a cat.",
        active_drifts=[],
    )

    assert "## 現在有効なSELF_DRIFT" not in prompt


def test_active_drifts_multiple_are_numbered():
    """active_drifts に複数件渡した場合、番号付きで列挙されること。

    システムプロンプト内で可読性を確保するため、
    drift 指針は「1. ...」「2. ...」のように番号付きリストになっていることを検証する。
    """
    drifts = ["一つ目の指針", "二つ目の指針", "三つ目の指針"]
    prompt = build_system_prompt(
        "You are a cat.",
        active_drifts=drifts,
    )

    # 番号付きリスト形式で各指針が含まれること
    assert "1. 一つ目の指針" in prompt
    assert "2. 二つ目の指針" in prompt
    assert "3. 三つ目の指針" in prompt
