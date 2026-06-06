"""allowed_tools 機能のテスト。

キャラクターごとの外部ツール許可設定 (web_search / google_calendar / gmail / google_drive) が
DB保存・ClaudeCliProvider の --tools フラグ生成・ChatRequest フィールドまで
正しく伝播することを検証する。
"""

import pytest

from backend.providers.claude_cli_provider import (
    ClaudeCliProvider,
    _build_cli_args,
    _build_tools_flag,
)
from backend.services.chat.models import ChatRequest, Message
from backend.api.utils import char_to_dict
from backend.api.schemas import CharacterCreate, CharacterUpdate


# ---------------------------------------------------------------------------
# _build_tools_flag
# ---------------------------------------------------------------------------

class TestBuildToolsFlag:
    """_build_tools_flag が allowed_tools 設定を --tools 文字列に正しく変換する。

    現状、組み込みツール（WebSearch/WebFetch 等）は一切有効化しない方針なので
    どの入力でも常に空文字列を返すことを検証する。外部情報の取得は Chotgor
    MCP の web_search ツール（Tavily 経由）に一本化されている。
    """

    def test_empty_dict_returns_empty_string(self):
        """設定なしは空文字列（全組み込みツール無効）。"""
        assert _build_tools_flag({}) == ""

    def test_all_false_returns_empty_string(self):
        """全ツール False でも空文字列。"""
        assert _build_tools_flag({
            "google_calendar": False,
            "gmail": False,
            "google_drive": False,
        }) == ""

    def test_google_tools_do_not_affect_tools_flag(self):
        """Google 系ツールは MCP 経由のため --tools フラグに反映されない。"""
        assert _build_tools_flag({"google_calendar": True}) == ""
        assert _build_tools_flag({"gmail": True}) == ""
        assert _build_tools_flag({"google_drive": True}) == ""

    def test_legacy_web_search_key_is_ignored(self):
        """旧 web_search キーが残っていても無視され、空文字列を返す（後方互換）。"""
        assert _build_tools_flag({"web_search": True}) == ""


# ---------------------------------------------------------------------------
# _build_cli_args
# ---------------------------------------------------------------------------

class TestBuildCliArgs:
    """_build_cli_args が --tools フラグを allowed_tools に基づいて組み立てる。"""

    def _get_tools_value(self, args: list[str]) -> str:
        """args リストから --tools の値を抽出するヘルパー。"""
        idx = args.index("--tools")
        return args[idx + 1]

    def test_no_allowed_tools_uses_empty_string(self):
        """allowed_tools 未指定は --tools "" （全組み込みツール無効）。"""
        args = _build_cli_args("sys")
        assert self._get_tools_value(args) == ""

    def test_none_allowed_tools_uses_empty_string(self):
        """allowed_tools=None も --tools ""。"""
        args = _build_cli_args("sys", allowed_tools=None)
        assert self._get_tools_value(args) == ""

    def test_legacy_web_search_key_does_not_enable_builtin_tools(self):
        """旧 web_search キーが立っていても --tools は空のまま（組み込みは廃止）。"""
        args = _build_cli_args("sys", allowed_tools={"web_search": True})
        assert self._get_tools_value(args) == ""

    def test_all_false_uses_empty_string(self):
        """全ツール False は --tools ""。"""
        args = _build_cli_args("sys", allowed_tools={
            "google_calendar": False,
            "gmail": False,
        })
        assert self._get_tools_value(args) == ""

    def test_model_flag_still_appended(self):
        """allowed_tools 追加後も --model フラグは正常に付与される。"""
        args = _build_cli_args("sys", model="claude-sonnet-4-6", allowed_tools={"google_calendar": True})
        assert "--model" in args
        assert "claude-sonnet-4-6" in args


# ---------------------------------------------------------------------------
# ClaudeCliProvider
# ---------------------------------------------------------------------------

class TestClaudeCliProviderAllowedTools:
    """ClaudeCliProvider が allowed_tools を受け取り保持する。"""

    def test_default_allowed_tools_is_empty_dict(self):
        """デフォルトは空 dict（全ツール無効）。"""
        provider = ClaudeCliProvider()
        assert provider.allowed_tools == {}

    def test_init_stores_allowed_tools(self):
        """__init__ に渡した allowed_tools が保持される。"""
        at = {"google_calendar": True, "gmail": False}
        provider = ClaudeCliProvider(allowed_tools=at)
        assert provider.allowed_tools == at

    def test_init_none_allowed_tools_coerces_to_empty(self):
        """None は空 dict に正規化される。"""
        provider = ClaudeCliProvider(allowed_tools=None)
        assert provider.allowed_tools == {}

    def test_from_config_stores_allowed_tools(self):
        """from_config 経由でも allowed_tools が保持される。"""
        at = {"google_calendar": True}
        provider = ClaudeCliProvider.from_config("", {}, allowed_tools=at)
        assert provider.allowed_tools == at

    def test_from_config_default_is_empty(self):
        """from_config で allowed_tools 未指定は空 dict。"""
        provider = ClaudeCliProvider.from_config("", {})
        assert provider.allowed_tools == {}


# ---------------------------------------------------------------------------
# SQLite CRUD（character_store）
# ---------------------------------------------------------------------------

class TestAllowedToolsSQLiteCRUD:
    """allowed_tools がキャラクターテーブルに正しく保存・取得される。"""

    def test_create_character_default_allowed_tools(self, sqlite_store):
        """allowed_tools 未指定で作成すると空 dict になる。"""
        char = sqlite_store.create_character("id1", "TestChar")
        assert char.allowed_tools == {}

    def test_create_character_with_allowed_tools(self, sqlite_store):
        """allowed_tools を指定して作成すると保持される。"""
        at = {"google_calendar": True, "gmail": False, "google_drive": False}
        char = sqlite_store.create_character("id2", "ToolChar", allowed_tools=at)
        assert char.allowed_tools == at

    def test_get_character_returns_allowed_tools(self, sqlite_store):
        """get_character で allowed_tools が取得できる。"""
        at = {"google_calendar": True}
        sqlite_store.create_character("id3", "ReadChar", allowed_tools=at)
        fetched = sqlite_store.get_character("id3")
        assert fetched.allowed_tools == at

    def test_update_character_allowed_tools(self, sqlite_store):
        """update_character で allowed_tools が更新される。"""
        sqlite_store.create_character("id4", "UpdateChar", allowed_tools={"google_calendar": False})
        sqlite_store.update_character("id4", allowed_tools={"google_calendar": True, "gmail": True})
        updated = sqlite_store.get_character("id4")
        assert updated.allowed_tools == {"google_calendar": True, "gmail": True}

    def test_create_without_allowed_tools_does_not_affect_other_fields(self, sqlite_store):
        """allowed_tools の追加が他フィールドを壊さない。"""
        char = sqlite_store.create_character(
            "id5", "SafeChar",
            system_prompt_block1="定義テキスト",
            allowed_tools={"google_calendar": True},
        )
        assert char.name == "SafeChar"
        assert char.system_prompt_block1 == "定義テキスト"
        assert char.allowed_tools == {"google_calendar": True}


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------

class TestChatRequestAllowedTools:
    """ChatRequest が allowed_tools フィールドを保持する。"""

    def _make_request(self, **kwargs) -> ChatRequest:
        defaults = dict(
            character_id="cid",
            character_name="Char",
            provider="claude_cli",
            model="",
            messages=[Message(role="user", content="hi")],
        )
        defaults.update(kwargs)
        return ChatRequest(**defaults)

    def test_default_allowed_tools_is_empty(self):
        """デフォルトは空 dict。"""
        req = self._make_request()
        assert req.allowed_tools == {}

    def test_allowed_tools_stored_correctly(self):
        """allowed_tools が正しく保持される。"""
        at = {"google_calendar": True, "gmail": False}
        req = self._make_request(allowed_tools=at)
        assert req.allowed_tools == at


# ---------------------------------------------------------------------------
# char_to_dict
# ---------------------------------------------------------------------------

class TestCharToDictAllowedTools:
    """char_to_dict が allowed_tools をレスポンスに含める。"""

    def test_allowed_tools_included_in_response(self, sqlite_store):
        """char_to_dict の出力に allowed_tools が含まれる。"""
        at = {"google_calendar": True, "gmail": False}
        char = sqlite_store.create_character("id_dict", "DictChar", allowed_tools=at)
        d = char_to_dict(char)
        assert "allowed_tools" in d
        assert d["allowed_tools"] == at

    def test_allowed_tools_defaults_to_empty_dict_in_response(self, sqlite_store):
        """allowed_tools 未設定キャラは空 dict が返る。"""
        char = sqlite_store.create_character("id_dict2", "EmptyChar")
        d = char_to_dict(char)
        assert d["allowed_tools"] == {}


# ---------------------------------------------------------------------------
# Pydantic スキーマ
# ---------------------------------------------------------------------------

class TestAllowedToolsSchemas:
    """CharacterCreate / CharacterUpdate の allowed_tools フィールド動作。"""

    def test_character_create_default_empty_dict(self):
        """CharacterCreate のデフォルトは空 dict。"""
        body = CharacterCreate(name="X")
        assert body.allowed_tools == {}

    def test_character_create_no_shared_mutable_default(self):
        """インスタンス間でデフォルト dict が共有されない。"""
        a = CharacterCreate(name="A")
        b = CharacterCreate(name="B")
        a.allowed_tools["google_calendar"] = True
        assert "google_calendar" not in b.allowed_tools

    def test_character_update_none_excluded_from_model_dump(self):
        """CharacterUpdate で allowed_tools 未指定時は None → update_character に渡らない。"""
        upd = CharacterUpdate(name="X")
        updates = {k: v for k, v in upd.model_dump().items() if v is not None}
        assert "allowed_tools" not in updates

    def test_character_update_explicit_value_included(self):
        """CharacterUpdate で allowed_tools を明示指定すると model_dump に含まれる。"""
        upd = CharacterUpdate(allowed_tools={"google_calendar": True})
        updates = {k: v for k, v in upd.model_dump().items() if v is not None}
        assert updates.get("allowed_tools") == {"google_calendar": True}
