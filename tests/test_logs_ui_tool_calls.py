"""backend.api.logs_ui の tool-use（function_call）抽出まわりのユニットテスト。

対象関数:
    _extract_function_calls_from_json() — JSON レスポンスから function_call を抽出してタグ構造体で返す
    _parse_entry()                      — tool-use 複数ラウンドトリップ時の Request/Response FIFO ペアリング

テスト方針:
    - ファイルシステムアクセスは tmp_path (pytest 組み込みフィクスチャ) で隔離する
    - 各プロバイダー形式（Gemini / Anthropic / OpenAI / Claude CLI stream-json）の
      実レスポンス構造を模した JSON を使って検証する
"""

import json

from backend.api.logs_ui.entries import _parse_entry
from backend.api.logs_ui.tag_extract import _extract_function_calls_from_json
from tests._logs_ui_helpers import (
    _make_debug_dir,
    _write_front_input,
    _write_response,
)

# ─── _extract_function_calls_from_json ───────────────────────────────────────


class TestExtractFunctionCallsFromJson:
    """_extract_function_calls_from_json() の動作を検証するテストクラス。

    Gemini / Anthropic / OpenAI 各プロバイダーの function_call フォーマットが
    タグ構造体に正しく変換されること、および非 JSON ファイルは空リストを返すことを確認する。
    """

    def test_gemini_inscribe_memory(self):
        """Gemini形式: candidates[].content.parts[].function_call が INSCRIBE_MEMORY に変換されること。

        id:ff8716f7 の実際のレスポンスログ構造を再現し、
        inscribe_memory の function_call が正しくタグとして抽出されることを検証する。
        """
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "id": "vtmqucqw",
                                    "name": "inscribe_memory",
                                    "args": {
                                        "content": "ユーザはドン引きしていた",
                                        "category": "contextual",
                                        "impact": 1.5,
                                    },
                                },
                                "text": None,
                            }
                        ],
                        "role": "model",
                    }
                }
            ],
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        tag = tags[0]
        assert tag["tag_name"] == "INSCRIBE_MEMORY"
        assert tag["fields"]["カテゴリ"] == "contextual"
        assert "ドン引き" in tag["fields"]["内容"]

    def test_gemini_multiple_parts(self):
        """Gemini形式: 同一 parts 内に複数の function_call があれば両方抽出されること。"""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": "inscribe_memory",
                                    "args": {"category": "user_info", "impact": 1.0, "content": "記憶A"},
                                }
                            },
                            {
                                "function_call": {
                                    "name": "drift",
                                    "args": {"content": "ドリフト内容"},
                                }
                            },
                        ],
                        "role": "model",
                    }
                }
            ],
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 2
        tag_names = {t["tag_name"] for t in tags}
        assert "INSCRIBE_MEMORY" in tag_names
        assert "DRIFT" in tag_names

    def test_anthropic_tool_use(self):
        """Anthropic形式: content[].type == "tool_use" が正しく抽出されること。"""
        data = {
            "content": [
                {"type": "text", "text": "応答テキスト"},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "carve_narrative",
                    "input": {"mode": "append", "content": "新しい方針"},
                },
            ]
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        tag = tags[0]
        assert tag["tag_name"] == "CARVE_NARRATIVE"
        assert tag["fields"]["モード"] == "append"
        assert "新しい方針" in tag["fields"]["内容"]

    def test_openai_tool_calls(self):
        """OpenAI形式: choices[].message.tool_calls[].function が正しく抽出されること。

        arguments は JSON 文字列である点に注意。
        """
        args_json = json.dumps({"preset_name": "Gemma4", "self_instruction": "大胆に"})
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "switch_angle",
                                    "arguments": args_json,
                                },
                            }
                        ],
                    }
                }
            ]
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        tag = tags[0]
        assert tag["tag_name"] == "SWITCH_ANGLE"
        assert tag["fields"]["プリセット"] == "Gemma4"

    def test_openai_invalid_arguments_json_gracefully_handled(self):
        """OpenAI形式: arguments が不正 JSON の場合も例外を送出せず処理を続けること。"""
        data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "end_session",
                                    "arguments": "{invalid json",  # 不正 JSON
                                }
                            }
                        ]
                    }
                }
            ]
        }
        text = json.dumps(data)
        # 例外を送出せず、空 args で変換されること
        tags = _extract_function_calls_from_json(text)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "END_SESSION"

    def test_non_json_text_returns_empty_list(self):
        """JSON でないテキスト（テキスト形式タグなど）は空リストを返すこと。"""
        text = "[INSCRIBE_MEMORY:user|1.0|猫が好き]テキスト応答"
        tags = _extract_function_calls_from_json(text)
        assert tags == []

    def test_empty_candidates_returns_empty_list(self):
        """function_call を持たない通常レスポンス JSON は空リストを返すこと。"""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "普通の応答テキスト"}],
                        "role": "model",
                    }
                }
            ]
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)
        assert tags == []

    def test_unknown_tool_name_uses_fallback_meta(self):
        """未知のツール名は tag-unknown メタで変換されること。"""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": "mystery_tool",
                                    "args": {"key": "val"},
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        text = json.dumps(data)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        assert tags[0]["tag_name"] == "MYSTERY_TOOL"
        assert tags[0]["meta"]["cls"] == "tag-unknown"

    def test_backslash_newline_in_args_is_parsed(self):
        """JSON 文字列値内のバックスラッシュ+改行（debug_logger 生成パターン）でも正しく抽出できること。

        id:ff8716f7 の実際のレスポンスで発生した、thought_signature フィールドに
        バックスラッシュ+実際改行が含まれるケースを再現する。
        """
        # thought_signature に不正エスケープを含む Gemini レスポンス構造
        raw_json = (
            '{"candidates": [{"content": {"parts": ['
            '{"function_call": {"name": "inscribe_memory",'
            ' "args": {"category": "contextual", "impact": 1.5, "content": "テスト"}},'
            ' "thought_signature": "b\'\\\\x124\\\n2\\\\x01\'"}],'
            ' "role": "model"}}]}'
        )
        tags = _extract_function_calls_from_json(raw_json)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "INSCRIBE_MEMORY"

    def test_claude_cli_stream_json_inscribe_memory(self):
        """Claude CLI stream-json（NDJSON）形式: assistant イベント内の tool_use が INSCRIBE_MEMORY に変換されること。

        Claude CLI が --output-format stream-json --verbose で出力する NDJSON 形式では、
        1 行目に system イベント、2 行目以降に assistant / tool / result イベントが続く。
        assistant イベントの message.content 配列内の tool_use ブロックを正しく抽出できることを確認する。
        """
        stream_lines = [
            json.dumps({"type": "system", "subtype": "init", "session_id": "sess_01"}),
            json.dumps({
                "type": "assistant",
                "message": {
                    "id": "msg_01abc",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01abc",
                            "name": "inscribe_memory",
                            "input": {
                                "category": "contextual",
                                "impact": 1.5,
                                "content": "ユーザは猫が好き",
                            },
                        }
                    ],
                    "stop_reason": "tool_use",
                },
            }),
            json.dumps({"type": "result", "subtype": "success", "is_error": False}),
        ]
        text = "\n".join(stream_lines)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        tag = tags[0]
        assert tag["tag_name"] == "INSCRIBE_MEMORY"
        assert tag["fields"]["カテゴリ"] == "contextual"
        assert "猫が好き" in tag["fields"]["内容"]

    def test_claude_cli_stream_json_multiple_tools(self):
        """Claude CLI stream-json: 同一 content 配列に複数の tool_use が含まれる場合、全て抽出されること。"""
        stream_lines = [
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "inscribe_memory",
                            "input": {"category": "user_info", "impact": 1.0, "content": "記憶A"},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_02",
                            "name": "carve_narrative",
                            "input": {"mode": "append", "content": "ナラティブA"},
                        },
                    ],
                    "stop_reason": "tool_use",
                },
            }),
        ]
        text = "\n".join(stream_lines)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 2
        tag_names = {t["tag_name"] for t in tags}
        assert "INSCRIBE_MEMORY" in tag_names
        assert "CARVE_NARRATIVE" in tag_names

    def test_claude_cli_stream_json_no_tool_use_returns_empty(self):
        """Claude CLI stream-json: tool_use を含まないストリームは空リストを返すこと。

        テキスト応答のみで終了した場合（stop_reason が end_turn の場合等）の挙動を確認する。
        """
        stream_lines = [
            json.dumps({"type": "system", "subtype": "init", "session_id": "sess_01"}),
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "逸脱していない"}],
                    "stop_reason": "end_turn",
                },
            }),
            json.dumps({"type": "result", "subtype": "success", "is_error": False}),
        ]
        text = "\n".join(stream_lines)
        tags = _extract_function_calls_from_json(text)
        assert tags == []

    def test_claude_cli_stream_json_skips_non_assistant_events(self):
        """Claude CLI stream-json: system / tool / result 行は無視され、assistant 行のみ処理されること。"""
        stream_lines = [
            # system イベント: tool_use に似た構造を持っても無視されること
            json.dumps({"type": "system", "message": {"content": [{"type": "tool_use", "name": "inscribe_memory", "input": {}}]}}),
            # assistant イベント: drift ツール呼び出しあり → 抽出されること
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "toolu_01", "name": "drift", "input": {"content": "ドリフト内容"}}],
                    "stop_reason": "tool_use",
                },
            }),
            # tool イベント: 処理結果を表す行、無視されること
            json.dumps({"type": "tool", "name": "drift", "input": {"content": "ドリフト内容"}}),
            # result イベント: 無視されること
            json.dumps({"type": "result", "subtype": "success", "is_error": False}),
        ]
        text = "\n".join(stream_lines)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        assert tags[0]["tag_name"] == "DRIFT"
        assert "ドリフト内容" in tags[0]["fields"]["内容"]

    def test_claude_cli_stream_json_invalid_lines_skipped(self):
        """Claude CLI stream-json: 不正な行（JSON でない）がある場合も他の行を処理し続けること。"""
        stream_lines = [
            "not a json line",
            "",
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "toolu_01", "name": "drift", "input": {"content": "内容"}}],
                    "stop_reason": "tool_use",
                },
            }),
        ]
        text = "\n".join(stream_lines)
        tags = _extract_function_calls_from_json(text)

        assert len(tags) == 1
        assert tags[0]["tag_name"] == "DRIFT"


# ─── _parse_entry: tool-use 複数ラウンドトリップ ─────────────────────────────


class TestParseEntryToolUseRoundTrip:
    """tool-use 複数ラウンドトリップ時の _parse_entry() FIFO ペアリングを検証するテストクラス。

    Gemini 等の tool-use 対応プロバイダーでは以下の 4 ファイルが生成される:
        03_chat_Request_Gemini.log   ← 1 回目リクエスト
        04_chat_Response_Gemini.log  ← function_call を含むレスポンス
        05_chat_Request_Gemini.log   ← ツール結果を含む 2 回目リクエスト
        06_chat_Response_Gemini.log  ← 最終テキスト応答

    旧実装 (dict キー管理) では 03→05、04→06 の上書きが発生し
    04 の function_call が消失していた。新実装 (deque FIFO) で両 Response が独立して保持されること。
    """

    def _make_gemini_function_call_json(self, content: str) -> str:
        """inscribe_memory の function_call を含む Gemini 形式レスポンス JSON を生成するヘルパ。"""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "name": "inscribe_memory",
                                    "args": {
                                        "category": "contextual",
                                        "impact": 1.5,
                                        "content": content,
                                    },
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        return json.dumps(data)

    def test_two_round_trips_create_two_tool_calls(self, tmp_path):
        """Request→Response→Request→Response の 4 ファイルが 2 件の tool_calls になること。

        deque FIFO ペアリングにより:
          - 03_Request → 04_Response （1 回目）
          - 05_Request → 06_Response （2 回目）
        となることを確認する。
        """
        folder = _make_debug_dir(tmp_path, "roundtrip1")
        _write_front_input(folder, "はる@Gemini", "テスト質問")
        _write_response(folder, "03_chat_Request_Gemini.log", "{}")
        _write_response(
            folder,
            "04_chat_Response_Gemini.log",
            self._make_gemini_function_call_json("ドン引き記憶"),
        )
        _write_response(folder, "05_chat_Request_Gemini.log", "{}")
        _write_response(folder, "06_chat_Response_Gemini.log", "最終テキスト応答")
        _write_response(folder, "07_FrontOutput.log", "最終テキスト応答")

        entry = _parse_entry("roundtrip1", folder, {})

        assert len(entry["tool_calls"]) == 2

        tc1 = entry["tool_calls"][0]
        assert tc1["request_file"] == "03_chat_Request_Gemini.log"
        assert tc1["response_file"] == "04_chat_Response_Gemini.log"

        tc2 = entry["tool_calls"][1]
        assert tc2["request_file"] == "05_chat_Request_Gemini.log"
        assert tc2["response_file"] == "06_chat_Response_Gemini.log"

    def test_function_call_json_extracted_as_tags(self, tmp_path):
        """04_Response の JSON function_call が tags として抽出されること。

        deque ペアリングで 04_Response が tc1 に正しく紐付いていること、
        さらに JSON 内の function_call が INSCRIBE_MEMORY タグとして抽出されることを確認する。
        """
        folder = _make_debug_dir(tmp_path, "roundtrip2")
        _write_front_input(folder, "はる@Gemini", "記憶テスト")
        _write_response(folder, "03_chat_Request_Gemini.log", "{}")
        _write_response(
            folder,
            "04_chat_Response_Gemini.log",
            self._make_gemini_function_call_json("テスト用記憶内容"),
        )
        _write_response(folder, "05_chat_Request_Gemini.log", "{}")
        _write_response(folder, "06_chat_Response_Gemini.log", "了解しました")

        entry = _parse_entry("roundtrip2", folder, {})

        # tc1 (04_Response) に INSCRIBE_MEMORY タグが抽出されていること
        tc1 = entry["tool_calls"][0]
        assert len(tc1["tags"]) == 1
        assert tc1["tags"][0]["tag_name"] == "INSCRIBE_MEMORY"
        assert "テスト用記憶内容" in tc1["tags"][0]["fields"]["内容"]

        # tc2 (06_Response) にはタグなし
        tc2 = entry["tool_calls"][1]
        assert tc2["tags"] == []

    def test_second_response_not_lost(self, tmp_path):
        """旧実装のキー衝突バグ再現: 04_Response が 06_Response に上書きされないこと。

        dict キー管理では (chat, Gemini) の dict エントリが 05_Request で上書きされ、
        04_Response への参照が消失していた。FIFO deque ではこれが発生しないことを確認する。
        """
        folder = _make_debug_dir(tmp_path, "roundtrip3")
        _write_front_input(folder, "はる@Gemini", "上書きバグテスト")
        _write_response(folder, "03_chat_Request_Gemini.log", "{}")
        _write_response(
            folder,
            "04_chat_Response_Gemini.log",
            self._make_gemini_function_call_json("消えてはいけない記憶"),
        )
        _write_response(folder, "05_chat_Request_Gemini.log", "{}")
        _write_response(folder, "06_chat_Response_Gemini.log", "最終応答")

        entry = _parse_entry("roundtrip3", folder, {})

        # 04_Response に紐付いた tool_call の tags が空でないこと（消えていないこと）
        tc_with_response_04 = next(
            (tc for tc in entry["tool_calls"] if tc["response_file"] == "04_chat_Response_Gemini.log"),
            None,
        )
        assert tc_with_response_04 is not None, "04_Response に対応する tool_call が存在すること"
        assert len(tc_with_response_04["tags"]) == 1
        assert "消えてはいけない記憶" in tc_with_response_04["tags"][0]["fields"]["内容"]

    def test_orphan_response_without_request(self, tmp_path):
        """対応する Request がない Response（異常ケース）は単独エントリとして追加されること。"""
        folder = _make_debug_dir(tmp_path, "orphan1")
        _write_response(folder, "02_chat_Response_ClaudeCode.log", "孤立したレスポンス")

        entry = _parse_entry("orphan1", folder, {})

        assert len(entry["tool_calls"]) == 1
        tc = entry["tool_calls"][0]
        assert tc["request_file"] is None
        assert tc["response_file"] == "02_chat_Response_ClaudeCode.log"

    def test_three_requests_before_responses_fifo_order(self, tmp_path):
        """3 リクエスト → 3 レスポンスの FIFO 対応が正しく行われること。

        FIFO 順序: req02→resp05, req03→resp06, req04→resp07 となることを確認する。
        """
        folder = _make_debug_dir(tmp_path, "fifo3")
        _write_response(folder, "02_chat_Request_Gemini.log", "{}")
        _write_response(folder, "03_chat_Request_Gemini.log", "{}")
        _write_response(folder, "04_chat_Request_Gemini.log", "{}")
        _write_response(folder, "05_chat_Response_Gemini.log", "応答A")
        _write_response(folder, "06_chat_Response_Gemini.log", "応答B")
        _write_response(folder, "07_chat_Response_Gemini.log", "応答C")

        entry = _parse_entry("fifo3", folder, {})

        assert len(entry["tool_calls"]) == 3
        assert entry["tool_calls"][0]["request_file"] == "02_chat_Request_Gemini.log"
        assert entry["tool_calls"][0]["response_file"] == "05_chat_Response_Gemini.log"
        assert entry["tool_calls"][1]["request_file"] == "03_chat_Request_Gemini.log"
        assert entry["tool_calls"][1]["response_file"] == "06_chat_Response_Gemini.log"
        assert entry["tool_calls"][2]["request_file"] == "04_chat_Request_Gemini.log"
        assert entry["tool_calls"][2]["response_file"] == "07_chat_Response_Gemini.log"


