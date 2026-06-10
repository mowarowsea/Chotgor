"""Tests for claude_cli_provider._format_conversation — Issue #10 coverage."""

import json

import pytest

from backend.providers.claude_cli_provider import (
    _extract_switch_angle_from_stream_json,
    _extract_usage_from_stream_json,
    _format_conversation,
)


class TestFormatConversation:
    def test_empty_messages(self):
        assert _format_conversation([]) == ""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "こんにちは"}]
        assert _format_conversation(messages) == "こんにちは"

    def test_single_user_message_with_char_name(self):
        messages = [{"role": "user", "content": "こんにちは"}]
        # 1件のみの場合は history なし、そのまま返す
        assert _format_conversation(messages, character_name="織羽") == "こんにちは"

    def test_multi_turn_uses_character_name_tag(self):
        """Issue #10: assistant ロールはキャラクター名タグで囲まれる。"""
        messages = [
            {"role": "user", "content": "調子はどう？"},
            {"role": "assistant", "content": "まあまあかな。"},
            {"role": "user", "content": "そっか"},
        ]
        result = _format_conversation(messages, character_name="織羽")
        expected = (
            "<history>\n"
            "<human>調子はどう？</human>\n"
            "<織羽>まあまあかな。</織羽>\n"
            "</history>\n\n"
            "そっか"
        )
        assert result == expected

    def test_multi_turn_fallback_when_no_character_name(self):
        """キャラクター名なし → <character> タグにフォールバック。"""
        messages = [
            {"role": "user", "content": "ねえ"},
            {"role": "assistant", "content": "なに？"},
            {"role": "user", "content": "なんでもない"},
        ]
        result = _format_conversation(messages)
        expected = (
            "<history>\n"
            "<human>ねえ</human>\n"
            "<character>なに？</character>\n"
            "</history>\n\n"
            "なんでもない"
        )
        assert result == expected

    def test_system_role_is_skipped(self):
        """system ロールはプロンプトフォーマットに含めない。"""
        messages = [
            {"role": "system", "content": "このメッセージは無視される"},
            {"role": "user", "content": "最初のユーザー発言"},
            {"role": "assistant", "content": "最初の返答"},
            {"role": "user", "content": "2番目の質問"},
        ]
        result = _format_conversation(messages, character_name="テスト")
        expected = (
            "<history>\n"
            "<human>最初のユーザー発言</human>\n"
            "<テスト>最初の返答</テスト>\n"
            "</history>\n\n"
            "2番目の質問"
        )
        assert result == expected

    def test_history_formatting(self):
        """historyタグの付与、humanタグの付与、キャラ名の余白除去をまとめてテスト。"""
        messages = [
            {"role": "user", "content": "過去の質問"},
            {"role": "assistant", "content": "過去の返答"},
            {"role": "user", "content": "最新の質問"},
        ]
        result = _format_conversation(messages, character_name="  織羽  ")
        expected = (
            "<history>\n"
            "<human>過去の質問</human>\n"
            "<織羽>過去の返答</織羽>\n"
            "</history>\n\n"
            "最新の質問"
        )
        assert result == expected


class TestMakeEnvBatchContext:
    """``ClaudeCliProvider._make_env(batch_context=...)`` のテスト。

    Claude CLI は subprocess として起動し、その中で MCP サーバ（別プロセス）が
    backend へ HTTP で問い合わせる構造のため、バッチ処理が指定する
    ``batch_context``（例: ``force_insert_memory=True``）は env 経由でしか伝搬できない。
    in-process プロバイダー（Ollama 等）では Python 側 ``ToolExecutor`` インスタンスが
    そのまま共有されるが、Claude CLI 経由では別プロセスへ復元する必要があるため、
    env→HTTP の経路を維持することが forget 蒸留バッチの正常動作（蒸留物の道連れ消失防止）に
    直結する。本テスト群はその伝搬経路の最小契約を守る。
    """

    def _make_provider(self):
        """テスト用のプロバイダー。character_id / session_id は固定値を持たせる。"""
        from backend.providers.claude_cli_provider import ClaudeCliProvider

        return ClaudeCliProvider(
            model="",
            character_name="はる",
            thinking_level="default",
            character_id="char-abc",
            session_id="sess-xyz",
        )

    def test_no_batch_context_omits_env_var(self):
        """``batch_context`` 未指定/None なら CHOTGOR_BATCH_CONTEXT は env に乗らないこと。

        通常 1on1 チャット時に余計な env が乗ってしまうと、MCP サーバ側が
        force_insert などのフラグを意図せず受け取ってしまう。OFF が確実に OFF であることを守る。
        """
        provider = self._make_provider()
        env = provider._make_env()
        assert "CHOTGOR_BATCH_CONTEXT" not in env
        # 既存のキャラ／セッション env はそのまま入る
        assert env["CHOTGOR_CHARACTER_ID"] == "char-abc"
        assert env["CHOTGOR_SESSION_ID"] == "sess-xyz"

    def test_empty_dict_batch_context_omits_env_var(self):
        """空 dict は「指定なし」と同義に扱われ、env に乗らないこと。

        ``ask_character_with_tools`` が ``batch_context=None`` 既定で呼ばれた際、
        途中経路で `{}` に正規化されても挙動が同じであることを保証する。
        """
        provider = self._make_provider()
        env = provider._make_env(batch_context={})
        assert "CHOTGOR_BATCH_CONTEXT" not in env

    def test_batch_context_serialized_as_json(self):
        """forget 蒸留の ``{"force_insert_memory": True}`` が JSON 文字列で env に乗ること。

        MCP サーバ側はこれを JSON parse して /api/mcp/tools/call の payload に
        そのまま転送するので、ここで JSON 表現が正しいことが伝搬経路全体の前提になる。
        """
        provider = self._make_provider()
        env = provider._make_env(batch_context={"force_insert_memory": True})
        raw = env["CHOTGOR_BATCH_CONTEXT"]
        assert json.loads(raw) == {"force_insert_memory": True}


class TestExtractSwitchAngleFromStreamJson:
    """Claude CLI の stream-json 出力から switch_angle ツール呼び出しを抽出する関数のテスト。

    Claude CLI は MCP サーバー（独立プロセス）経由で switch_angle を実行するため、
    ChatService 側の tool_executor.switch_request には自動反映されない。
    raw stream-json に含まれる mcp__chotgor__switch_angle の tool_use ブロックから
    (preset_name, self_instruction) を抽出して、generate_with_tools が
    tool_executor に転写できるようにする必要がある。

    本クラスは抽出関数の挙動を検証する：
    - 正常系: switch_angle 呼び出しが含まれる stream-json から正しく抽出できる
    - 異常系: tool_use が無い／別ツールの呼び出しのみ／JSON 不正行が混じる、等の場合 None
    - 入力欠落系: preset_name が無い場合は None、self_instruction は空文字許容
    - 複数件: 複数呼び出しがあるときは最初の1件のみ返す（switch は1ターンに1回）
    """

    def _make_assistant_event(self, content_blocks: list[dict]) -> str:
        """assistant ロールの stream-json イベント1行を組み立てるヘルパー。"""
        return json.dumps({
            "type": "assistant",
            "message": {"content": content_blocks},
        }, ensure_ascii=False)

    def test_extract_switch_angle_basic(self):
        """mcp__chotgor__switch_angle 呼び出しを含む stream-json から (preset_name, self_instruction) が抽出される。"""
        raw = self._make_assistant_event([
            {
                "type": "tool_use",
                "name": "mcp__chotgor__switch_angle",
                "input": {
                    "preset_name": "Gemini3_1FlashLite",
                    "self_instruction": "軽くさっぱりと",
                },
            }
        ])
        assert _extract_switch_angle_from_stream_json(raw) == (
            "Gemini3_1FlashLite", "軽くさっぱりと"
        )

    def test_extract_returns_none_when_no_tool_use(self):
        """tool_use ブロックが無い（text のみ）応答からは None が返る。"""
        raw = self._make_assistant_event([
            {"type": "text", "text": "ただのテキスト応答"}
        ])
        assert _extract_switch_angle_from_stream_json(raw) is None

    def test_extract_returns_none_for_other_tools(self):
        """switch_angle 以外のツール呼び出しのみのときは None が返る。"""
        raw = self._make_assistant_event([
            {
                "type": "tool_use",
                "name": "mcp__chotgor__inscribe_memory",
                "input": {"content": "test", "category": "user", "impact": 1.0},
            }
        ])
        assert _extract_switch_angle_from_stream_json(raw) is None

    def test_extract_ignores_invalid_json_lines(self):
        """JSON として無効な行が混じっていてもパースを中断せず後続行を見つけられる。"""
        valid = self._make_assistant_event([
            {
                "type": "tool_use",
                "name": "mcp__chotgor__switch_angle",
                "input": {"preset_name": "preset-a", "self_instruction": "instr"},
            }
        ])
        raw = "{壊れたJSON\n" + valid
        assert _extract_switch_angle_from_stream_json(raw) == ("preset-a", "instr")

    def test_extract_returns_none_when_preset_name_missing(self):
        """preset_name が欠けている tool_use は None を返す（実行不能なので無視）。"""
        raw = self._make_assistant_event([
            {
                "type": "tool_use",
                "name": "mcp__chotgor__switch_angle",
                "input": {"self_instruction": "instr のみ"},
            }
        ])
        assert _extract_switch_angle_from_stream_json(raw) is None

    def test_extract_allows_empty_self_instruction(self):
        """self_instruction が省略されていても preset_name さえあれば抽出される。"""
        raw = self._make_assistant_event([
            {
                "type": "tool_use",
                "name": "mcp__chotgor__switch_angle",
                "input": {"preset_name": "preset-x"},
            }
        ])
        assert _extract_switch_angle_from_stream_json(raw) == ("preset-x", "")

    def test_extract_returns_first_call_only(self):
        """同一応答内に複数 switch_angle 呼び出しがあるときは最初の1件だけ返す。"""
        first = self._make_assistant_event([
            {
                "type": "tool_use",
                "name": "mcp__chotgor__switch_angle",
                "input": {"preset_name": "first", "self_instruction": "a"},
            }
        ])
        second = self._make_assistant_event([
            {
                "type": "tool_use",
                "name": "mcp__chotgor__switch_angle",
                "input": {"preset_name": "second", "self_instruction": "b"},
            }
        ])
        raw = first + "\n" + second
        assert _extract_switch_angle_from_stream_json(raw) == ("first", "a")

    def test_extract_works_with_text_and_tool_use_mixed(self):
        """1つの assistant メッセージ内に text と tool_use が混在しても tool_use を拾える。

        実際の Claude CLI 出力（debug ログ）と同型のパターン。
        """
        raw = self._make_assistant_event([
            {"type": "thinking", "thinking": "切り替えよう"},
            {"type": "text", "text": "行ってみよ。"},
            {
                "type": "tool_use",
                "name": "mcp__chotgor__switch_angle",
                "input": {"preset_name": "fastModel", "self_instruction": "軽く"},
            },
        ])
        assert _extract_switch_angle_from_stream_json(raw) == ("fastModel", "軽く")


class TestExtractUsageFromStreamJson:
    """Claude CLI の stream-json 出力からトークン使用量を抽出する関数のテスト。

    使用量はダッシュボード（/ui/）の集計の元データになるため、抽出の正確さと
    「取れないときは記録しない（None）」の両方を確認する：
    - 正常系: result イベントの usage / total_cost_usd と assistant イベントの
      message.model（実際に使われたモデルID）が揃って抽出される
    - 欠落系: result イベントが無い（途中エラー等）場合は None
    - 部分欠落系: usage のキー欠け・cost 無しはゼロ／None で補完される
    - 汚染系: 非 JSON 行が混ざっても result イベントを取りこぼさない
    - 複数件: result が複数あるときは最後の1件（最終的な合計）を採用する
    """

    def _make_result_event(self, usage: dict, cost: float | None = None) -> str:
        """result イベント1行を組み立てるヘルパー。"""
        event: dict = {"type": "result", "usage": usage}
        if cost is not None:
            event["total_cost_usd"] = cost
        return json.dumps(event, ensure_ascii=False)

    def test_extract_full_usage(self):
        """usage 全項目・cost・モデルIDが揃った stream-json から全て抽出されること。"""
        raw = (
            json.dumps({
                "type": "assistant",
                "message": {"model": "claude-sonnet-4-6", "content": []},
            })
            + "\n"
            + self._make_result_event(
                {
                    "input_tokens": 1200,
                    "output_tokens": 340,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 50,
                },
                cost=0.0123,
            )
        )

        assert _extract_usage_from_stream_json(raw) == {
            "model": "claude-sonnet-4-6",
            "input_tokens": 1200,
            "output_tokens": 340,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 50,
            "total_cost_usd": 0.0123,
        }

    def test_returns_none_without_result_event(self):
        """result イベントが無い（途中エラー等で打ち切られた）場合は None が返ること。"""
        raw = json.dumps({
            "type": "assistant",
            "message": {"model": "claude-sonnet-4-6", "content": []},
        })

        assert _extract_usage_from_stream_json(raw) is None

    def test_missing_usage_keys_default_to_zero(self):
        """usage のキーが一部欠けていてもゼロ補完され、cost 無しは None になること。"""
        raw = self._make_result_event({"input_tokens": 10})

        extracted = _extract_usage_from_stream_json(raw)

        assert extracted == {
            "model": "",
            "input_tokens": 10,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "total_cost_usd": None,
        }

    def test_garbage_lines_do_not_swallow_result(self):
        """非 JSON 行が混ざっても後続の result イベントを取りこぼさないこと。"""
        raw = "WARNING: cli notice\n" + self._make_result_event(
            {"input_tokens": 7, "output_tokens": 3}
        )

        extracted = _extract_usage_from_stream_json(raw)

        assert extracted is not None
        assert extracted["input_tokens"] == 7
        assert extracted["output_tokens"] == 3

    def test_last_result_event_wins(self):
        """result イベントが複数ある場合は最後の1件（最終合計）が採用されること。"""
        raw = (
            self._make_result_event({"input_tokens": 1, "output_tokens": 1}, cost=0.001)
            + "\n"
            + self._make_result_event({"input_tokens": 100, "output_tokens": 50}, cost=0.02)
        )

        extracted = _extract_usage_from_stream_json(raw)

        assert extracted is not None
        assert extracted["input_tokens"] == 100
        assert extracted["output_tokens"] == 50
        assert extracted["total_cost_usd"] == 0.02

    def test_first_assistant_model_wins(self):
        """assistant イベントが複数あるときは最初の model を採用すること（tool-use ループでも同一実行）。"""
        raw = (
            json.dumps({"type": "assistant", "message": {"model": "model-a", "content": []}})
            + "\n"
            + json.dumps({"type": "assistant", "message": {"model": "model-b", "content": []}})
            + "\n"
            + self._make_result_event({"input_tokens": 1, "output_tokens": 1})
        )

        extracted = _extract_usage_from_stream_json(raw)

        assert extracted is not None
        assert extracted["model"] == "model-a"
