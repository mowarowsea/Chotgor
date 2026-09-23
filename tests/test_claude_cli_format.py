"""Tests for claude_cli_provider._format_conversation — Issue #10 coverage."""

import json

from backend.providers.claude_cli_provider import (
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


class TestFormatConversationSpeakerTags:
    """1on1（user_label あり）で「タグ＝話者」に揃える整形のテスト。

    旧形式では履歴のユーザが `<human>` と抽象化され、最新発言は `</history>` の後ろに
    裸で置かれ、その直後にタグ無しのターン注釈（キャラ一人称の想起記憶を含む）が
    地続きで続いていた。このため、はるが自分の発言をもわの発言と取り違える事故が
    少なくない頻度で起きていた（2026-09-23）。ここでは次の契約を守る:
    - 履歴・最新発言ともユーザ側を user_label のタグで包む
    - 最新発言末尾のターン注釈（<turn_context>）はユーザタグの外に出す
    - 最新ターンがターン注釈だけ（発話予約の単独発火）ならユーザタグを付けない
      （Chotgor 由来の文脈をユーザ発言として見せると、逆に誤帰属になる）
    - user_label 未指定の経路（バッチ問い合わせ）は旧形式のまま
    """

    def test_history_and_latest_use_user_label(self):
        """履歴も最新発言も <もわ> で包まれ、<human> が現れないこと。"""
        messages = [
            {"role": "user", "content": "ユーザの発言だよ"},
            {"role": "assistant", "content": "知ってる"},
            {"role": "user", "content": "最後の返信だよ"},
        ]
        result = _format_conversation(messages, "はる", "もわ")
        assert result == (
            "<history>\n"
            "<もわ>ユーザの発言だよ</もわ>\n"
            "<はる>知ってる</はる>\n"
            "</history>\n\n"
            "<もわ>最後の返信だよ</もわ>"
        )
        assert "<human>" not in result

    def test_turn_context_is_placed_outside_user_tag(self):
        """注釈付きの最新発言は、発言だけが <もわ> に入り注釈はその外に出ること。"""
        messages = [
            {"role": "user", "content": "前の話"},
            {"role": "assistant", "content": "うん"},
            {
                "role": "user",
                "content": "最後の返信だよ\n\n<turn_context>\n【このターンの文脈】\n記憶\n</turn_context>",
            },
        ]
        result = _format_conversation(messages, "はる", "もわ")
        assert result.endswith(
            "</history>\n\n"
            "<もわ>最後の返信だよ</もわ>\n\n"
            "<turn_context>\n【このターンの文脈】\n記憶\n</turn_context>"
        )

    def test_context_only_turn_has_no_user_tag(self):
        """最新ターンが注釈だけ（発話予約の単独発火）なら <もわ> を付けないこと。

        合成注釈と通常のターン注釈の2ブロックが並んでも、両方ともタグの外に残る。
        """
        latest = (
            "<turn_context>\n（いま 10:00。あなたから声をかける番）\n</turn_context>\n\n"
            "<turn_context>\n【このターンの文脈】\n</turn_context>"
        )
        messages = [
            {"role": "user", "content": "おやすみ"},
            {"role": "assistant", "content": "おやすみ"},
            {"role": "user", "content": latest},
        ]
        result = _format_conversation(messages, "はる", "もわ")
        assert result.endswith("</history>\n\n" + latest)
        assert result.count("<もわ>") == 1  # 履歴側の1件のみ

    def test_single_message_is_wrapped_with_user_label(self):
        """セッション最初の発言（履歴なし）も <もわ> で包まれること。"""
        messages = [{"role": "user", "content": "はじめまして"}]
        assert _format_conversation(messages, "はる", "もわ") == "<もわ>はじめまして</もわ>"

    def test_multipart_latest_with_annotation_part(self):
        """画像添付などマルチパートでも、注釈パートがタグの外に出ること。

        append_turn_annotation はマルチパート時に注釈を別テキストパートとして足すため、
        パートの連結後に分割できることを確認する。
        """
        messages = [
            {"role": "user", "content": "前"},
            {"role": "assistant", "content": "うん"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "これ見て"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "\n\n<turn_context>\n文脈\n</turn_context>"},
                ],
            },
        ]
        result = _format_conversation(messages, "はる", "もわ")
        assert result.endswith("<もわ>これ見て</もわ>\n\n<turn_context>\n文脈\n</turn_context>")

    def test_without_user_label_keeps_legacy_format(self):
        """user_label 未指定なら旧形式（<human>・最新は素通し）のままであること。

        バッチ問い合わせ（character_query）の user ロールはユーザ本人ではなく
        Chotgor からの問い合わせなので、ユーザ名で包んではならない。
        """
        messages = [
            {"role": "user", "content": "問い"},
            {"role": "assistant", "content": "答え"},
            {"role": "user", "content": "次の問い\n\n<turn_context>\n文脈\n</turn_context>"},
        ]
        result = _format_conversation(messages, "はる")
        assert "<human>問い</human>" in result
        assert result.endswith("</history>\n\n次の問い\n\n<turn_context>\n文脈\n</turn_context>")

    def test_turn_context_tag_matches_request_builder(self):
        """分割に使うタグ名が request_builder の注釈タグと一致していること。

        providers→services の import 逆流を避けるため値を複製しているので、
        片方だけ改名されると分割が効かなくなり注釈がユーザ発言に取り込まれる。
        """
        from backend.providers.claude_cli_provider import _TURN_CONTEXT_OPEN
        from backend.services.chat.request_builder import TURN_CONTEXT_OPEN

        assert _TURN_CONTEXT_OPEN == TURN_CONTEXT_OPEN


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


class TestMcpEnabledFlag:
    """「ツール無しの問い合わせでは MCP を閉じる」契約のテスト。

    Chotgor MCP サーバはユーザグローバルの ``~/.claude.json`` に登録されているため、
    CLI をどの cwd で起動しても接続され、ツール一覧がキャラクターへ提示される。
    一方 ``generate()`` 経路（ask_character・ambience judge・計器判定・GM 予定生成・翻訳）は
    provider へ character_id を渡さないため、MCP サーバ側で
    ``[Error: CHOTGOR_CHARACTER_ID が設定されていません]`` になり、ツールは必ず失敗する。
    実際に Chronicle でキャラクターがスレッド更新ツールを6回叩いて全滅し、
    JSON 出力へ自力リカバリするという事故が起きた（debug/2a46c2cb）。

    そこで ``generate()`` は ``--strict-mcp-config``（--mcp-config 未指定と組み合わせて
    MCP サーバ 0 本）で起動し、使えない手を最初から見せない。逆に ``generate_with_tools()``
    は MCP ループ本体なので、絶対に閉じてはならない。本テスト群はこの左右の振り分けを守る。
    """

    def _args(self, **kwargs):
        from backend.providers.claude_cli_provider import _build_cli_args

        return _build_cli_args("sys prompt", **kwargs)

    def test_default_keeps_mcp_connected(self):
        """既定（mcp_enabled 省略）では --strict-mcp-config を付けないこと。

        tool-use ループ（generate_with_tools）はこの既定で動く。ここに紛れ込むと
        記憶ツールが丸ごと使えなくなり、キャラクターが記憶を残せなくなる。
        """
        assert "--strict-mcp-config" not in self._args()

    def test_disabled_adds_strict_flag_without_mcp_config(self):
        """mcp_enabled=False で --strict-mcp-config が付き、--mcp-config は付かないこと。

        「--mcp-config を渡さずに --strict-mcp-config を付ける」の組み合わせが
        サーバー 0 本の条件そのものなので、両方をまとめて検証する。
        """
        args = self._args(mcp_enabled=False)
        assert "--strict-mcp-config" in args
        assert "--mcp-config" not in args

    async def test_generate_closes_mcp(self, monkeypatch):
        """generate() が mcp_enabled=False で CLI を起動すること。"""
        from backend.providers.claude_cli_provider import ClaudeCliProvider

        captured = {}

        async def fake_raw(self, system_prompt, messages, **kwargs):
            captured.update(kwargs)
            return ""

        monkeypatch.setattr(ClaudeCliProvider, "_run_generate_raw", fake_raw)
        await ClaudeCliProvider(character_name="はる").generate("sys", [{"role": "user", "content": "hi"}])
        assert captured.get("mcp_enabled") is False

    async def test_generate_with_tools_keeps_mcp(self, monkeypatch):
        """generate_with_tools() は MCP を閉じないこと（mcp_enabled を落とさない）。"""
        from backend.providers.claude_cli_provider import ClaudeCliProvider

        captured = {}

        async def fake_raw(self, system_prompt, messages, **kwargs):
            captured.update(kwargs)
            return ""

        monkeypatch.setattr(ClaudeCliProvider, "_run_generate_raw", fake_raw)
        await ClaudeCliProvider(character_name="はる", character_id="char-abc").generate_with_tools(
            "sys", [{"role": "user", "content": "hi"}], None,
        )
        assert captured.get("mcp_enabled", True) is True


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
