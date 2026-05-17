"""シナリオチャット 「あらすじ」自動要約モジュールのテスト。

backend.services.scenario_chat.synopsis の純関数部分を中心に検証する。
update_auto_synopsis 自体は LLM プロバイダー呼出を含むため、本テストでは
ダミープロバイダで E2E ライクに最小確認のみ行う。

検証する観点:
    - `append_auto_synopsis` の追記ロジック（上書きしない、既存空ならそのまま、改行区切り）
    - `build_synopsis_system_prompt` の主要要素（既存 auto の埋め込み・書き換え禁止文言）
    - `update_auto_synopsis` が dropped 空・preset 不在・LLM 出力空 などのエッジで None を返すこと
    - `update_auto_synopsis` が成功時に既存 auto + LLM 出力を返すこと（既存を破壊しない）

記憶捏造対策の核心は「ユーザ手編集の保護」なので、追記の不変条件を厚く検証する。
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

import pytest

from backend.services.scenario_chat.synopsis import (
    append_auto_synopsis,
    build_synopsis_system_prompt,
    update_auto_synopsis,
)


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


@dataclass
class FakeScenario:
    """ZetaScenario 風のダミーオブジェクト。"""

    user_alias: str = "プレイヤー"
    gm_preset_id: str = "preset-001"
    scenario: Optional[str] = None


@dataclass
class FakeTurn:
    """ZetaTurn 風のダミーオブジェクト。"""

    speaker_type: str
    speaker_name: str
    content: str
    turn_index: int = 0


@dataclass
class FakePreset:
    """LLMModelPreset 風のダミー。"""

    id: str = "preset-001"
    name: str = "TestPreset"
    provider: str = "fake"
    model_id: str = "fake-model"


class FakeProvider:
    """generate_stream_typed() で固定の (type, chunk) 列を返すスタブ。"""

    def __init__(self, chunks: list[tuple[str, str]], raises: Optional[Exception] = None):
        """
        Args:
            chunks: yield する (type, content) ペアのリスト。
            raises: None でなければ generate_stream_typed の呼び出しで送出する例外。
        """
        self.chunks = chunks
        self.raises = raises
        self.received_system_prompt: Optional[str] = None

    async def generate_stream_typed(self, system_prompt: str, messages: list[dict]):
        """テスト用の固定出力をストリーミング風に返す。"""
        self.received_system_prompt = system_prompt
        if self.raises is not None:
            raise self.raises
        for typ, content in self.chunks:
            yield typ, content


# ─── append_auto_synopsis（追記の不変条件） ─────────────────────────────────


class TestAppendAutoSynopsis:
    """既存 synopsis_auto への追記ロジックを検証する。

    記憶捏造対策の核心は「ユーザが手編集した auto の既存記述を
    自動更新が破壊しない」こと。本クラスはその不変条件を厚く検証する。
    """

    def test_append_to_empty_returns_new_only(self):
        """既存が空なら新規要約だけが返ること。"""
        result = append_auto_synopsis("", "新しいあらすじ")
        assert result == "新しいあらすじ"

    def test_append_preserves_existing_text(self):
        """既存テキストが先頭に、改行区切りで新規が末尾に来ること。"""
        existing = "勇者は森でレイカと出会った。"
        new = "二人は北の村へ向かった。"
        result = append_auto_synopsis(existing, new)
        assert existing in result
        assert new in result
        # 既存が先に来る
        assert result.index(existing) < result.index(new)

    def test_append_separates_with_blank_line(self):
        """既存と新規の間に空行（\\n\\n）が入ること。"""
        result = append_auto_synopsis("既存", "新規")
        assert "既存\n\n新規" == result

    def test_append_empty_new_returns_existing_unchanged(self):
        """新規が空なら既存をそのまま返すこと（破壊しない）。"""
        existing = "守りたい既存テキスト"
        assert append_auto_synopsis(existing, "") == existing
        assert append_auto_synopsis(existing, "   \n  ") == existing

    def test_append_strips_trailing_whitespace_from_existing(self):
        """既存末尾の余分な空白・改行は除去された上で連結されること。"""
        result = append_auto_synopsis("既存\n\n\n", "新規")
        assert result == "既存\n\n新規"

    def test_append_strips_new_summary_whitespace(self):
        """新規要約の前後余分空白は除去されること。"""
        result = append_auto_synopsis("既存", "  \n新規\n  ")
        assert result == "既存\n\n新規"

    def test_user_edited_text_is_preserved(self):
        """ユーザが手編集した既存 auto が、追記後も完全に保持されること（核心）。"""
        user_edited = (
            "（プレイヤー注：レイカは実は王女である）\n"
            "勇者は森でレイカと出会った。"
        )
        new_auto_summary = "二人は王城へ向かった。"
        result = append_auto_synopsis(user_edited, new_auto_summary)
        # ユーザ編集部分が一字一句保持されている
        assert "（プレイヤー注：レイカは実は王女である）" in result
        assert "勇者は森でレイカと出会った。" in result
        # 新規分が末尾に追加されている
        assert result.endswith("二人は王城へ向かった。")


# ─── build_synopsis_system_prompt（プロンプト構造） ─────────────────────────


class TestBuildSynopsisSystemPrompt:
    """要約 LLM 用 system prompt の構造を検証する。"""

    def test_includes_user_alias_as_protagonist(self):
        """プロンプトに主役（user_alias）が含まれること。"""
        out = build_synopsis_system_prompt(FakeScenario(user_alias="勇者"), "")
        assert "@勇者" in out

    def test_existing_auto_embedded(self):
        """既存 auto が「これまでのあらすじ」セクションとして埋め込まれること。"""
        out = build_synopsis_system_prompt(
            FakeScenario(), existing_auto="勇者はレイカと出会った。"
        )
        assert "勇者はレイカと出会った。" in out

    def test_existing_auto_marked_as_unchangeable(self):
        """既存 auto が「書き換え禁止・繰り返し禁止」と明示されること。"""
        out = build_synopsis_system_prompt(
            FakeScenario(), existing_auto="既存のあらすじ。"
        )
        assert "書き換え禁止" in out or "書き換え" in out

    def test_empty_existing_omits_section(self):
        """既存 auto が空なら、その関連セクションは出ないこと。"""
        out = build_synopsis_system_prompt(FakeScenario(), existing_auto="")
        assert "書き換え禁止" not in out

    def test_scenario_text_included_when_present(self):
        """シナリオ世界観テキストがあれば含まれること。"""
        out = build_synopsis_system_prompt(
            FakeScenario(scenario="ここは雨の街。"), existing_auto=""
        )
        assert "ここは雨の街。" in out


# ─── update_auto_synopsis（end-to-end with mock provider） ──────────────────


def _loader_for(preset: Optional[FakePreset]):
    """preset_loader 関数を作る（preset が None なら常に None を返す）。"""
    def loader(preset_id: str):
        return preset
    return loader


def _factory_for(provider: FakeProvider):
    """provider_factory 関数を作る。"""
    def factory(provider_id: str, model: str, settings: dict, **kwargs):
        return provider
    return factory


class TestUpdateAutoSynopsis:
    """`update_auto_synopsis` の振る舞いを mock プロバイダで検証する。

    主観点:
        - dropped_turns 空 → None
        - preset_loader が None → None
        - LLM 出力が空 → None
        - 通常成功時に既存 auto + 新規要約 が返ること
        - 既存 auto が要約 system prompt に含まれること
    """

    @pytest.mark.asyncio
    async def test_empty_dropped_returns_none(self):
        """dropped_turns が空なら何もせず None を返すこと。"""
        provider = FakeProvider(chunks=[("text", "要約")])
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            dropped_turns=[],
            existing_auto="既存",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            provider_factory=_factory_for(provider),
        )
        assert result is None
        # プロバイダは呼ばれていない
        assert provider.received_system_prompt is None

    @pytest.mark.asyncio
    async def test_missing_preset_returns_none(self):
        """preset_loader が None を返したら None。"""
        provider = FakeProvider(chunks=[("text", "要約")])
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            dropped_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="やぁ")
            ],
            existing_auto="",
            settings={},
            preset_loader=_loader_for(None),
            provider_factory=_factory_for(provider),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_llm_output_returns_none(self):
        """LLM 出力が空文字列のみなら None を返すこと（既存を破壊しない）。"""
        provider = FakeProvider(chunks=[("text", "   "), ("text", "")])
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            dropped_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="やぁ")
            ],
            existing_auto="既存テキスト",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            provider_factory=_factory_for(provider),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_summary_appended_to_existing(self):
        """通常成功時、既存 auto + LLM 要約 が返ること。"""
        provider = FakeProvider(chunks=[("text", "新ターンの要約結果。")])
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            dropped_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A"),
                FakeTurn(speaker_type="narrator", speaker_name="N", content="B"),
            ],
            existing_auto="昔の出来事。",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            provider_factory=_factory_for(provider),
        )
        assert result == "昔の出来事。\n\n新ターンの要約結果。"

    @pytest.mark.asyncio
    async def test_existing_auto_in_system_prompt(self):
        """既存 auto が要約用 system prompt に「書き換え禁止」として埋め込まれること。"""
        provider = FakeProvider(chunks=[("text", "x")])
        await update_auto_synopsis(
            scenario=FakeScenario(),
            dropped_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A")
            ],
            existing_auto="守りたい既存記述",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            provider_factory=_factory_for(provider),
        )
        assert provider.received_system_prompt is not None
        assert "守りたい既存記述" in provider.received_system_prompt

    @pytest.mark.asyncio
    async def test_provider_exception_returns_none(self):
        """LLM 呼出で例外が起きたら None を返し既存を破壊しないこと。"""
        provider = FakeProvider(chunks=[], raises=RuntimeError("LLM 故障"))
        result = await update_auto_synopsis(
            scenario=FakeScenario(),
            dropped_turns=[
                FakeTurn(speaker_type="user", speaker_name="P", content="A")
            ],
            existing_auto="守りたい既存",
            settings={},
            preset_loader=_loader_for(FakePreset()),
            provider_factory=_factory_for(provider),
        )
        assert result is None
