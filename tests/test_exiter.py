"""backend.core.chat.exiter モジュールのユニットテスト。

Exiter クラスのタグ方式（exit_from_text）・ツール呼び出し方式（set_exit）・
定数エクスポート・ToolExecutor との統合を検証する。
"""

import pytest
from unittest.mock import MagicMock

from backend.core.chat.exiter import (
    Exiter,
    END_SESSION_SCHEMA,
    END_SESSION_TAG_GUIDE,
    END_SESSION_TOOL_DESCRIPTION,
)


# ─── 定数エクスポートテスト ───────────────────────────────────────────────────

class TestConstants:
    """定数が正しくエクスポートされているかを検証する。"""

    def test_tag_guide_is_nonempty_string(self):
        """END_SESSION_TAG_GUIDE が空でない文字列である。"""
        assert isinstance(END_SESSION_TAG_GUIDE, str)
        assert len(END_SESSION_TAG_GUIDE) > 0

    def test_tool_description_is_nonempty_string(self):
        """END_SESSION_TOOL_DESCRIPTION が空でない文字列である。"""
        assert isinstance(END_SESSION_TOOL_DESCRIPTION, str)
        assert len(END_SESSION_TOOL_DESCRIPTION) > 0

    def test_schema_is_valid_json_schema(self):
        """END_SESSION_SCHEMA が type=object の JSON スキーマ形式である。"""
        assert END_SESSION_SCHEMA["type"] == "object"
        assert "properties" in END_SESSION_SCHEMA
        # reason は optional なので required は空リストまたは存在しない
        required = END_SESSION_SCHEMA.get("required", [])
        assert "reason" not in required

    def test_tag_guide_contains_end_session_tag(self):
        """END_SESSION_TAG_GUIDE に [END_SESSION:...] の記述が含まれる。"""
        assert "END_SESSION" in END_SESSION_TAG_GUIDE


# ─── Exiter 初期状態テスト ────────────────────────────────────────────────────

class TestExiterInit:
    """Exiter の初期状態が正しいかを検証する。"""

    def test_exit_reason_is_none_initially(self):
        """初期状態では exit_reason が None である。"""
        exiter = Exiter()
        assert exiter.exit_reason is None

    def test_has_exit_is_false_initially(self):
        """初期状態では has_exit が False である。"""
        exiter = Exiter()
        assert exiter.has_exit is False


# ─── exit_from_text (タグ方式) ────────────────────────────────────────────────

class TestExitFromText:
    """exit_from_text メソッドのタグ検出・除去を検証する。"""

    def test_reason_tag_detected(self):
        """[END_SESSION:reason] タグを正しく検出する。"""
        exiter = Exiter()
        text = "お疲れ様でした。[END_SESSION:疲れたので今日はここまで]"
        clean = exiter.exit_from_text(text)
        assert exiter.has_exit
        assert exiter.exit_reason == "疲れたので今日はここまで"

    def test_reason_tag_removed_from_clean_text(self):
        """[END_SESSION:reason] タグがクリーンテキストから除去される。"""
        exiter = Exiter()
        text = "またね。[END_SESSION:用事があります]"
        clean = exiter.exit_from_text(text)
        assert "END_SESSION" not in clean
        assert "またね。" in clean

    def test_empty_reason_colon_form(self):
        """[END_SESSION:] 形式（reason 省略・コロンあり）で reason が空文字列になる。"""
        exiter = Exiter()
        text = "さようなら[END_SESSION:]"
        clean = exiter.exit_from_text(text)
        assert exiter.has_exit
        assert exiter.exit_reason == ""

    def test_fixed_marker_form(self):
        """[END_SESSION] 固定マーカー形式（コロンなし）で reason が空文字列になる。"""
        exiter = Exiter()
        text = "じゃあね。[END_SESSION]"
        clean = exiter.exit_from_text(text)
        assert exiter.has_exit
        assert exiter.exit_reason == ""

    def test_no_tag_returns_unchanged(self):
        """タグがない場合はテキストが変更されず、has_exit が False のまま。"""
        exiter = Exiter()
        text = "普通の会話です。"
        clean = exiter.exit_from_text(text)
        assert clean == text
        assert not exiter.has_exit
        assert exiter.exit_reason is None

    def test_tag_in_code_block_ignored(self):
        """バッククォートコードブロック内のタグは無視される。"""
        exiter = Exiter()
        text = "例えば `[END_SESSION:test]` と書くとこうなります。"
        clean = exiter.exit_from_text(text)
        # コードブロック内は無視されるため退席しない
        assert not exiter.has_exit

    def test_multiple_tags_uses_first(self):
        """複数タグがある場合は最初のタグが使用される。"""
        exiter = Exiter()
        text = "goodbye[END_SESSION:first][END_SESSION:second]"
        clean = exiter.exit_from_text(text)
        assert exiter.exit_reason == "first"

    def test_exit_reason_trimmed(self):
        """退席理由の前後の空白がトリムされる。"""
        exiter = Exiter()
        text = "[END_SESSION:  スペースあり  ]"
        exiter.exit_from_text(text)
        assert exiter.exit_reason == "スペースあり"

    def test_text_before_tag_preserved(self):
        """タグ前のテキストがクリーンテキストに保持される。"""
        exiter = Exiter()
        text = "これは通常の返答です。[END_SESSION:帰ります]"
        clean = exiter.exit_from_text(text)
        assert "これは通常の返答です" in clean

    def test_idempotent_on_repeated_calls(self):
        """タグなしテキストを繰り返し呼んでも has_exit が変化しない。"""
        exiter = Exiter()
        for _ in range(3):
            exiter.exit_from_text("普通のテキスト")
        assert not exiter.has_exit


# ─── set_exit (ツール呼び出し方式) ───────────────────────────────────────────

class TestSetExit:
    """set_exit メソッドのツール呼び出し方式を検証する。"""

    def test_set_exit_with_reason(self):
        """reason を指定して set_exit を呼ぶと exit_reason が設定される。"""
        exiter = Exiter()
        result = exiter.set_exit(reason="疲れた")
        assert exiter.has_exit
        assert exiter.exit_reason == "疲れた"
        assert isinstance(result, str)

    def test_set_exit_without_reason(self):
        """reason を省略して set_exit を呼ぶと exit_reason が空文字列になる。"""
        exiter = Exiter()
        exiter.set_exit()
        assert exiter.has_exit
        assert exiter.exit_reason == ""

    def test_set_exit_empty_string_reason(self):
        """reason="" を指定した場合は空文字列になる。"""
        exiter = Exiter()
        exiter.set_exit(reason="")
        assert exiter.exit_reason == ""

    def test_set_exit_reason_stripped(self):
        """reason の前後空白がトリムされる。"""
        exiter = Exiter()
        exiter.set_exit(reason="  スペース  ")
        assert exiter.exit_reason == "スペース"

    def test_set_exit_returns_string(self):
        """set_exit の戻り値が文字列である。"""
        exiter = Exiter()
        result = exiter.set_exit(reason="test")
        assert isinstance(result, str)


# ─── ToolExecutor 統合テスト ──────────────────────────────────────────────────

class TestToolExecutorEndSession:
    """ToolExecutor が end_session ツールを正しく処理するかを検証する。"""

    def _make_executor(self):
        """テスト用 ToolExecutor を生成するヘルパー。"""
        from backend.core.tools import ToolExecutor
        memory_manager = MagicMock()
        memory_manager.sqlite = MagicMock()
        drift_manager = MagicMock()
        return ToolExecutor(
            character_id="char-001",
            session_id="session-001",
            memory_manager=memory_manager,
            drift_manager=drift_manager,
        )

    def test_end_session_tool_sets_exit_reason(self):
        """end_session ツールを実行すると exit_reason が設定される。"""
        executor = self._make_executor()
        executor.execute("end_session", {"reason": "疲れた"})
        assert executor.exit_reason == "疲れた"

    def test_end_session_tool_without_reason(self):
        """reason なしの end_session ツールで exit_reason が空文字列になる。"""
        executor = self._make_executor()
        executor.execute("end_session", {})
        assert executor.exit_reason == ""

    def test_exit_reason_none_before_tool_call(self):
        """end_session ツール呼び出し前は exit_reason が None である。"""
        executor = self._make_executor()
        assert executor.exit_reason is None

    def test_end_session_in_anthropic_tools(self):
        """ANTHROPIC_TOOLS に end_session が含まれる。"""
        from backend.core.tools import ANTHROPIC_TOOLS
        names = [t["name"] for t in ANTHROPIC_TOOLS]
        assert "end_session" in names

    def test_end_session_in_openai_tools(self):
        """OPENAI_TOOLS に end_session が含まれる。"""
        from backend.core.tools import OPENAI_TOOLS
        names = [t["function"]["name"] for t in OPENAI_TOOLS]
        assert "end_session" in names


# ─── system_prompt 統合テスト ─────────────────────────────────────────────────

class TestSystemPromptIntegration:
    """system_prompt に END_SESSION ガイドが含まれるかを検証する。"""

    def test_end_session_guide_in_tag_mode_prompt(self):
        """タグ方式のシステムプロンプトに END_SESSION のガイドが含まれる。"""
        from backend.core.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            character_system_prompt="テストキャラです。",
            use_tools=False,
        )
        assert "END_SESSION" in prompt

    def test_end_session_mentioned_in_tools_mode_prompt(self):
        """ツール方式のシステムプロンプトに end_session の言及が含まれる。"""
        from backend.core.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            character_system_prompt="テストキャラです。",
            use_tools=True,
        )
        assert "end_session" in prompt


# ─── DBスキーマ統合テスト ─────────────────────────────────────────────────────

class TestDatabaseSchema:
    """新規カラムが正しく作成・利用できるかを検証する。"""

    def test_chat_session_has_exited_chars_column(self, sqlite_store):
        """chat_sessions テーブルに exited_chars カラムが存在する。"""
        session = sqlite_store.create_chat_session(
            session_id="s001",
            model_id="Alice@preset1",
        )
        # exited_chars は初期値 None
        assert getattr(session, "exited_chars", "MISSING") is None

    def test_update_exited_chars(self, sqlite_store):
        """exited_chars を更新できる。"""
        sqlite_store.create_chat_session(session_id="s002", model_id="Alice@preset1")
        exited = [{"char_name": "Alice", "reason": "疲れた"}]
        updated = sqlite_store.update_chat_session("s002", exited_chars=exited)
        assert updated.exited_chars == exited

    def test_chat_message_has_is_system_message_column(self, sqlite_store):
        """chat_messages テーブルに is_system_message カラムが存在する。"""
        sqlite_store.create_chat_session(session_id="s003", model_id="Alice@preset1")
        msg = sqlite_store.create_chat_message(
            message_id="m001",
            session_id="s003",
            role="character",
            content="退席します",
            is_system_message=True,
        )
        assert getattr(msg, "is_system_message", "MISSING") == 1

    def test_normal_message_has_no_system_flag(self, sqlite_store):
        """通常メッセージの is_system_message が None である。"""
        sqlite_store.create_chat_session(session_id="s004", model_id="Alice@preset1")
        msg = sqlite_store.create_chat_message(
            message_id="m002",
            session_id="s004",
            role="character",
            content="こんにちは",
        )
        assert getattr(msg, "is_system_message", "MISSING") is None


# ─── chat.py ヘルパー関数テスト ───────────────────────────────────────────────

class TestChatApiHelpers:
    """chat.py の退席メッセージ生成ヘルパーを検証する。"""

    def test_build_exit_message_with_reason(self):
        """reason ありの退席メッセージが正しく生成される。"""
        from backend.api.chat import _build_exit_message
        msg = _build_exit_message("Alice", "疲れた")
        assert "Alice" in msg
        assert "疲れた" in msg
        assert "退席" in msg

    def test_build_exit_message_without_reason(self):
        """reason なしの退席メッセージが正しく生成される。"""
        from backend.api.chat import _build_exit_message
        msg = _build_exit_message("Bob", "")
        assert "Bob" in msg
        assert "退席" in msg
        # 理由の文字が含まれないこと
        assert "理由:" not in msg

    def test_build_all_exited_message_multiple_chars(self):
        """複数キャラクターの全員退席メッセージが全員分含まれる。"""
        from backend.api.chat import _build_all_exited_message
        exited = [
            {"char_name": "Alice", "reason": "疲れた"},
            {"char_name": "Bob", "reason": ""},
        ]
        msg = _build_all_exited_message(exited)
        assert "Alice" in msg
        assert "Bob" in msg
        assert "疲れた" in msg

    def test_build_all_exited_message_contains_guidance(self):
        """全員退席メッセージに新規セッション案内が含まれる。"""
        from backend.api.chat import _build_all_exited_message
        exited = [{"char_name": "Alice", "reason": ""}]
        msg = _build_all_exited_message(exited)
        assert "新しいセッション" in msg


# ─── utils.py message_to_dict テスト ─────────────────────────────────────────

class TestMessageToDict:
    """message_to_dict が is_system_message を正しく扱うかを検証する。"""

    def test_system_message_flag_included(self):
        """is_system_message=1 のメッセージでフラグが True として含まれる。"""
        from backend.api.utils import message_to_dict
        m = MagicMock()
        m.id = "msg-001"
        m.session_id = "sess-001"
        m.role = "character"
        m.content = "退席します"
        m.created_at = None
        m.reasoning = None
        m.images = None
        m.character_name = None
        m.preset_name = None
        m.is_system_message = 1
        result = message_to_dict(m)
        assert result.get("is_system_message") is True

    def test_normal_message_flag_omitted(self):
        """is_system_message が None の通常メッセージではフラグが省略される。"""
        from backend.api.utils import message_to_dict
        m = MagicMock()
        m.id = "msg-002"
        m.session_id = "sess-001"
        m.role = "character"
        m.content = "こんにちは"
        m.created_at = None
        m.reasoning = None
        m.images = None
        m.character_name = None
        m.preset_name = None
        m.is_system_message = None
        result = message_to_dict(m)
        assert "is_system_message" not in result


# ─── utils.py session_to_dict テスト ─────────────────────────────────────────

class TestSessionToDict:
    """session_to_dict が exited_chars を正しく扱うかを検証する。"""

    def test_exited_chars_included_when_set(self):
        """exited_chars が設定されている場合はレスポンスに含まれる。"""
        from backend.api.utils import session_to_dict
        s = MagicMock()
        s.id = "sess-001"
        s.model_id = "Alice@preset1"
        s.title = "テスト"
        s.session_type = "1on1"
        s.created_at = None
        s.updated_at = None
        s.group_config = None
        s.afterglow_session_id = None
        s.exited_chars = [{"char_name": "Alice", "reason": "疲れた"}]
        result = session_to_dict(s)
        assert "exited_chars" in result
        assert result["exited_chars"][0]["char_name"] == "Alice"

    def test_exited_chars_omitted_when_none(self):
        """exited_chars が None の場合はレスポンスに含まれない。"""
        from backend.api.utils import session_to_dict
        s = MagicMock()
        s.id = "sess-002"
        s.model_id = "Alice@preset1"
        s.title = "テスト"
        s.session_type = "1on1"
        s.created_at = None
        s.updated_at = None
        s.group_config = None
        s.afterglow_session_id = None
        s.exited_chars = None
        result = session_to_dict(s)
        assert "exited_chars" not in result
