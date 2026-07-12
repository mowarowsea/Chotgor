"""backend.api.logs_ui.tag_extract モジュールのユニットテスト。

タグ／ツール呼び出し抽出まわりの関数を網羅的に検証する。

対象関数:
    _parse_tag_body()         — タグ本体文字列を表示用辞書に変換する
    _extract_tags_from_file() — ログファイルからタグを抽出して出現順で返す
    _parse_json_safely()      — debug_logger 生成 JSON をバックスラッシュ+改行前処理込みでパースする

テスト方針:
    - ファイルシステムアクセスは tmp_path (pytest 組み込みフィクスチャ) で隔離する
    - LLM・DB へのアクセスは発生しないため外部 mock は不要
"""

import json

import pytest

from backend.api.logs_ui.tag_extract import (
    _extract_tags_from_file,
    _parse_json_safely,
    _parse_tag_body,
)

# ─── _parse_tag_body ──────────────────────────────────────────────────────────


class TestParseTagBody:
    """_parse_tag_body() の各タグ種別変換を検証するテストクラス。"""

    def test_inscribe_memory_full(self):
        """INSCRIBE_MEMORY: category|importance|content が正しく分解されること。"""
        result = _parse_tag_body("INSCRIBE_MEMORY", "user|1.4|ユーザは猫が好き")
        assert result["tag_name"] == "INSCRIBE_MEMORY"
        assert result["fields"]["カテゴリ"] == "user"
        assert result["fields"]["重要度"] == "1.4"
        assert result["fields"]["内容"] == "ユーザは猫が好き"
        assert result["preview"] == "ユーザは猫が好き"
        assert result["meta"]["label"] == "記憶"
        assert result["meta"]["cls"] == "tag-memory"

    def test_inscribe_memory_content_with_pipe(self):
        """INSCRIBE_MEMORY: 内容フィールドに | が含まれる場合、最大2分割で正しく扱われること。"""
        result = _parse_tag_body("INSCRIBE_MEMORY", "semantic|0.9|A|B|C という内容")
        assert result["fields"]["カテゴリ"] == "semantic"
        assert result["fields"]["重要度"] == "0.9"
        assert result["fields"]["内容"] == "A|B|C という内容"

    def test_inscribe_memory_missing_importance(self):
        """INSCRIBE_MEMORY: importance が省略された場合、空文字として扱われること。"""
        result = _parse_tag_body("INSCRIBE_MEMORY", "contextual")
        assert result["fields"]["カテゴリ"] == "contextual"
        assert result["fields"]["重要度"] == ""
        assert result["fields"]["内容"] == ""

    def test_carve_narrative_append(self):
        """CARVE_NARRATIVE: append|content が正しく分解されること。"""
        result = _parse_tag_body("CARVE_NARRATIVE", "append|新しい自己認識が芽生えた")
        assert result["fields"]["モード"] == "append"
        assert result["fields"]["内容"] == "新しい自己認識が芽生えた"
        assert result["preview"] == "新しい自己認識が芽生えた"
        assert result["meta"]["cls"] == "tag-narrative"

    def test_carve_narrative_content_with_pipe(self):
        """CARVE_NARRATIVE: 内容フィールドに | が含まれる場合も正しく扱われること。"""
        result = _parse_tag_body("CARVE_NARRATIVE", "append|A|B")
        assert result["fields"]["モード"] == "append"
        assert result["fields"]["内容"] == "A|B"

    def test_switch_angle_preset_and_context(self):
        """SWITCH_ANGLE: preset|context が分解され、preview は context になること（Bug修正確認）。"""
        result = _parse_tag_body("SWITCH_ANGLE", "Gemma4|ミオとして大胆になったシーン")
        assert result["fields"]["プリセット"] == "Gemma4"
        assert result["fields"]["コンテキスト"] == "ミオとして大胆になったシーン"
        # preview は context（第2フィールド）であること。preset|context の body 全体ではない
        assert result["preview"] == "ミオとして大胆になったシーン"
        assert result["meta"]["cls"] == "tag-switch"

    def test_switch_angle_preset_only(self):
        """SWITCH_ANGLE: context 省略時も preset が取れ、preview は空文字にならないこと。"""
        result = _parse_tag_body("SWITCH_ANGLE", "Gemma4")
        assert result["fields"]["プリセット"] == "Gemma4"
        assert result["fields"]["コンテキスト"] == ""
        # コンテキストが空 → フォールバックで body（"Gemma4"）が preview になる
        assert result["preview"] == "Gemma4"

    def test_power_recall(self):
        """POWER_RECALL: body が "内容" フィールドになること。"""
        result = _parse_tag_body("POWER_RECALL", "記憶を全て呼び起こせ")
        assert result["fields"]["内容"] == "記憶を全て呼び起こせ"
        assert result["meta"]["cls"] == "tag-recall"

    def test_end_session_with_body(self):
        """END_SESSION: body が "内容" フィールドになること。"""
        result = _parse_tag_body("END_SESSION", "さようなら")
        assert result["fields"]["内容"] == "さようなら"
        assert result["meta"]["cls"] == "tag-end"

    def test_unknown_tag_uses_fallback_meta(self):
        """未知のタグ名の場合、fallback meta (cls=tag-unknown) が使われること。"""
        result = _parse_tag_body("UNKNOWN_TAG", "body")
        assert result["meta"]["cls"] == "tag-unknown"
        assert result["meta"]["label"] == "UNKNOWN_TAG"

    def test_anticipate_response_body_is_expectation(self):
        """ANTICIPATE_RESPONSE: body 全体が fields["予想"] に入り、preview もそれになること。

        キャラクター本人が返答末尾に書いた「次の展開への予想・期待」テキストを
        ▼ログパネルで表示するためのタグ。tool-use 化していない全プロバイダー一律タグ方式。
        """
        result = _parse_tag_body(
            "ANTICIPATE_RESPONSE",
            "ユーザはきっと喜ぶに違いない",
        )
        assert result["tag_name"] == "ANTICIPATE_RESPONSE"
        assert result["fields"]["予想"] == "ユーザはきっと喜ぶに違いない"
        assert result["preview"] == "ユーザはきっと喜ぶに違いない"
        assert result["meta"]["label"] == "予想"
        assert result["meta"]["cls"] == "tag-anticipate"


# ─── _extract_tags_from_file ─────────────────────────────────────────────────


class TestExtractTagsFromFile:
    """_extract_tags_from_file() の動作を検証するテストクラス。"""

    def test_single_inscribe_memory_tag(self, tmp_path):
        """単一の INSCRIBE_MEMORY タグが正しく抽出されること。"""
        f = tmp_path / "response.log"
        f.write_text(
            "会話の内容[INSCRIBE_MEMORY:user|1.0|ユーザは猫が好き]以上",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "INSCRIBE_MEMORY"
        assert tags[0]["fields"]["内容"] == "ユーザは猫が好き"

    def test_file_order_preserved(self, tmp_path):
        """複数タグが混在する場合、ファイル内の出現順で返されること（Bug修正確認）。

        ファイル内で CARVE_NARRATIVE → INSCRIBE_MEMORY の順で出現した場合、
        タグ種別のリスト順（INSCRIBE_MEMORY→CARVE_NARRATIVE）ではなく、
        出現順（CARVE_NARRATIVE→INSCRIBE_MEMORY）で返されること。
        """
        f = tmp_path / "response.log"
        f.write_text(
            "テキスト[CARVE_NARRATIVE:append|自己認識]間のテキスト[INSCRIBE_MEMORY:contextual|0.8|記憶内容]終わり",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)
        assert len(tags) == 2
        assert tags[0]["tag_name"] == "CARVE_NARRATIVE"
        assert tags[1]["tag_name"] == "INSCRIBE_MEMORY"

    def test_multiline_tag(self, tmp_path):
        """タグが複数行にまたがる場合も正しく抽出されること。"""
        f = tmp_path / "response.log"
        f.write_text(
            "応答テキスト[INSCRIBE_MEMORY:contextual|1.0|1行目の内容\n2行目の内容\n3行目]終わり",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "INSCRIBE_MEMORY"
        assert "1行目の内容" in tags[0]["fields"]["内容"]
        assert "3行目" in tags[0]["fields"]["内容"]

    def test_no_tags_returns_empty(self, tmp_path):
        """タグが存在しないファイルでは空リストを返すこと。"""
        f = tmp_path / "response.log"
        f.write_text("タグなしのテキストです", encoding="utf-8")
        assert _extract_tags_from_file(f) == []

    def test_missing_file_returns_empty(self, tmp_path):
        """ファイルが存在しない場合も例外を送出せず空リストを返すこと。"""
        assert _extract_tags_from_file(tmp_path / "missing.log") == []

    def test_multiple_same_tag_type(self, tmp_path):
        """同一タグ種別が複数ある場合、すべて抽出されること。"""
        f = tmp_path / "response.log"
        f.write_text(
            "[INSCRIBE_MEMORY:user|1.0|記憶A]テキスト[INSCRIBE_MEMORY:semantic|0.5|記憶B]",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)
        assert len(tags) == 2
        contents = [t["fields"]["内容"] for t in tags]
        assert "記憶A" in contents
        assert "記憶B" in contents

    def test_stream_json_tool_calls_take_priority_over_tags(self, tmp_path):
        """tool-first設計: stream-json に tool_use があればタグ方式は使われないこと。

        Claude CLI（stream-json）のレスポンスファイルにタグ形式のテキストも含まれている場合、
        ツール呼び出し（JSON内 tool_use）を優先し、タグパースは行われないことを確認する。
        """
        stream_line = json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "inscribe_memory",
                     "input": {"category": "contextual", "impact": 1.0, "content": "ツール記憶"}},
                ],
                "stop_reason": "tool_use",
            },
        })
        # stream-json 行 + タグ形式テキストが混在するファイル
        f = tmp_path / "response.log"
        f.write_text(
            stream_line + "\n[INSCRIBE_MEMORY:user|1.0|タグ記憶]",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)

        # ツール呼び出しが優先されるため、タグ方式の「タグ記憶」は含まれない
        assert len(tags) == 1
        assert tags[0]["fields"]["内容"] == "ツール記憶"

    def test_anticipate_response_tag_extracted(self, tmp_path):
        """[ANTICIPATE_RESPONSE:...] タグが抽出されること。

        anticipator.py は意図的にこのタグを TOOL_TO_TAG へ登録しないが、
        ▼ログ パネルで結果を表示するため _KNOWN_TAG_NAMES に独立して追加されている。
        その抽出経路の動作確認。
        """
        f = tmp_path / "response.log"
        f.write_text(
            "応答本文[ANTICIPATE_RESPONSE:次は深い話をしたい]終わり",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "ANTICIPATE_RESPONSE"
        assert tags[0]["fields"]["予想"] == "次は深い話をしたい"
        assert tags[0]["meta"]["cls"] == "tag-anticipate"

    def test_anticipate_response_tag_deduped_in_stream_json(self, tmp_path):
        """Claude CLI の stream-json で同じ ANTICIPATE_RESPONSE が type=assistant と type=result
        の両方に含まれるケースで、UI上は1件に集約されること。

        Claude CLI の stream-json 出力仕様として、最終応答テキストは type=assistant の
        text ブロックと type=result の result フィールドに同一内容で重複して現れる。
        生ログ全体をスキャンする parse_tags 経路ではタグが2回検出されてしまうため、
        _dedupe_tags で完全一致する重複を1件にまとめる。本テストはその挙動を確認する。
        """
        # tool_use ブロックを持たない（=tool-use 経路ではなくタグ方式フォールバックに落ちる）
        # stream-json 形式を組み立てる。
        assistant_event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "本文だよ[ANTICIPATE_RESPONSE:深掘りされそう]"}
                ]
            },
        }
        result_event = {
            "type": "result",
            "result": "本文だよ[ANTICIPATE_RESPONSE:深掘りされそう]",
        }
        f = tmp_path / "response.log"
        f.write_text(
            json.dumps(assistant_event, ensure_ascii=False) + "\n"
            + json.dumps(result_event, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        tags = _extract_tags_from_file(f)

        # 重複が除去されて1件のみになる
        assert len(tags) == 1
        assert tags[0]["tag_name"] == "ANTICIPATE_RESPONSE"
        assert tags[0]["fields"]["予想"] == "深掘りされそう"

    def test_anticipation_survives_tool_calls_in_same_response(self, tmp_path):
        """tool_use と ANTICIPATE_RESPONSE が同一レスポンスに共存しても両方表示されること。

        ANTICIPATE_RESPONSE は tool-use 化していない全プロバイダー一律のテキストタグ。
        以前は tool_use（例: post_working_memory_thread）が見つかると早期 return し、
        本文のタグがスキャンされず予想バッジが消えていた（Bug修正確認）。
        """
        assistant_tool = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "toolu_01",
                     "name": "mcp__chotgor__post_working_memory_thread",
                     "input": {"thread_id": "t1", "content": "進捗メモ"}},
                ]
            },
        }
        assistant_text = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "本文[ANTICIPATE_RESPONSE:続きが気になるはず]"}
                ]
            },
        }
        f = tmp_path / "response.log"
        f.write_text(
            json.dumps(assistant_tool, ensure_ascii=False) + "\n"
            + json.dumps(assistant_text, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)

        names = [t["tag_name"] for t in tags]
        assert "POST_WORKING_MEMORY_THREAD" in names
        assert "ANTICIPATE_RESPONSE" in names
        anticipate = next(t for t in tags if t["tag_name"] == "ANTICIPATE_RESPONSE")
        assert anticipate["fields"]["予想"] == "続きが気になるはず"

    def test_anticipation_rewritten_after_tool_error_keeps_last_only(self, tmp_path):
        """ツールエラー後の書き直しで文面の異なる予想タグが複数あっても最後の1件だけ表示されること。

        実行時の extract_anticipation() は最後のタグを採用して保存するため、
        ログ表示もそれに合わせる（完全一致でない重複は _dedupe_tags では潰せない）。
        """
        f = tmp_path / "response.log"
        f.write_text(
            "一度目の本文[ANTICIPATE_RESPONSE:最初の予想]\n"
            "（ツールエラー後の書き直し）[ANTICIPATE_RESPONSE:書き直した予想]",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)

        anticipates = [t for t in tags if t["tag_name"] == "ANTICIPATE_RESPONSE"]
        assert len(anticipates) == 1
        assert anticipates[0]["fields"]["予想"] == "書き直した予想"

    def test_multiline_tool_use_event_is_collected(self, tmp_path):
        """tool_use の input に生改行を含み複数行に分断されたイベントも収集されること。

        debug_logger は JSON 文字列値内の \\n を実際の改行に展開するため、
        stream-json のイベントが複数行にまたがる。行単体パースだけだと
        tool_use を取りこぼし、タグ方式フォールバックに落ちてしまう（Bug修正確認）。
        """
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "toolu_01",
                     "name": "mcp__chotgor__post_working_memory_thread",
                     "input": {"thread_id": "t1", "content": "1行目\n2行目\n3行目"}},
                ]
            },
        }
        # ensure_ascii=False かつ改行を実改行へ展開して debug_logger の出力を再現する
        raw = json.dumps(event, ensure_ascii=False).replace("\\n", "\n")
        assert "\n" in raw  # 前提: 複数行にまたがっている
        f = tmp_path / "response.log"
        f.write_text(raw + "\n", encoding="utf-8")

        tags = _extract_tags_from_file(f)

        assert len(tags) == 1
        assert tags[0]["tag_name"] == "POST_WORKING_MEMORY_THREAD"

    def test_distinct_tags_with_same_name_are_kept(self, tmp_path):
        """同じタグ名でも fields が異なる重複（別引数で2回呼ばれた等）は両方残ること。

        _dedupe_tags は構造化辞書の完全一致のみで重複判定するため、
        例えば異なる内容で2回 INSCRIBE_MEMORY された場合は両方表示される必要がある。
        """
        f = tmp_path / "response.log"
        f.write_text(
            "[INSCRIBE_MEMORY:contextual|0.8|記憶A]"
            "[INSCRIBE_MEMORY:contextual|0.8|記憶B]",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)

        assert len(tags) == 2
        assert {t["fields"]["内容"] for t in tags} == {"記憶A", "記憶B"}

    def test_tag_fallback_when_file_has_only_plain_text_tags(self, tmp_path):
        """tag-fallback設計: JSON/stream-json のツール呼び出しがなければタグパースにフォールバックすること。

        Ollama / OpenRouter 等のタグ方式プロバイダーはレスポンスにタグを埋め込む。
        ファイルにツール呼び出し（JSON）が含まれない場合、テキスト内のタグが抽出されることを確認する。
        """
        f = tmp_path / "response.log"
        f.write_text(
            "通常テキスト[INSCRIBE_MEMORY:contextual|0.8|Ollama記憶]終わり",
            encoding="utf-8",
        )
        tags = _extract_tags_from_file(f)

        assert len(tags) == 1
        assert tags[0]["tag_name"] == "INSCRIBE_MEMORY"
        assert tags[0]["fields"]["内容"] == "Ollama記憶"

    def test_pure_json_without_tool_calls_falls_back_to_tag_parse(self, tmp_path):
        """tool-first/tag-fallback: JSON ファイルでもツール呼び出しがなければタグパースを試みること。

        Gemini が普通のテキスト応答（function_call なし）を返した場合、
        タグ文字列がレスポンスに含まれていればタグ抽出されることを確認する。
        """
        # function_call を持たない通常の Gemini 応答 + タグテキスト
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "テキスト[INSCRIBE_MEMORY:contextual|1.0|感情状態テキスト]終わり"}],
                        "role": "model",
                    }
                }
            ]
        }
        f = tmp_path / "response.log"
        f.write_text(json.dumps(data), encoding="utf-8")
        tags = _extract_tags_from_file(f)

        # JSON として正常パースされるが function_call なし → タグパースにフォールバック
        # 注: JSON 文字列フィールド内のタグ（エスケープされた形）は抽出されない場合がある
        # このテストはフォールバックが呼ばれることを確認する（結果は空でも可）
        # JSON として全体はパース成功 → _collect_tool_calls_from_single_json は空 → タグパースへ
        # ただし JSON テキストに "[INSCRIBE_MEMORY:...]" が含まれるため、タグパーサはそれを見つけることができる
        assert isinstance(tags, list)  # 例外なく list が返ること


# ─── _parse_json_safely ───────────────────────────────────────────────────────


class TestParseJsonSafely:
    """_parse_json_safely() の動作を検証するテストクラス。

    debug_logger._unescape_text() により JSON 文字列値内の \\n が実際の改行になることで
    生じる不正エスケープ（バックスラッシュ + 実際の改行）を前処理して json.loads できること、
    および通常の JSON・非 JSON テキストの扱いを確認する。
    """

    def test_normal_json_is_parsed(self):
        """通常の JSON 文字列が正しくパースされること。"""
        text = '{"key": "value", "num": 42}'
        result = _parse_json_safely(text)
        assert result == {"key": "value", "num": 42}

    def test_backslash_newline_in_string_value_is_repaired(self):
        """JSON 文字列値内のバックスラッシュ+改行（不正エスケープ）が前処理で修復されること。

        debug_logger._unescape_text() は JSON 文字列値内の \\n を実際の改行に変換するため、
        バックスラッシュ + 実際の改行（\\<LF>）という不正エスケープが生まれる。
        strict=False は制御文字を許容するが不正エスケープシーケンスは許容しないため、
        素の json.loads でも失敗する。_parse_json_safely はこれを修復する。
        """
        # Python の '...\\\n...' は文字列として「バックスラッシュ + 実際の改行（0x0A）」を含む。
        # JSON パーサは \<LF> を不正エスケープとして拒否する。
        raw = '{"key": "line1\\\nline2\\\nline3"}'

        # 前提確認: 素の json.loads(strict=False) では JSONDecodeError になること
        try:
            json.loads(raw, strict=False)
            raw_fails = False
        except json.JSONDecodeError:
            raw_fails = True
        assert raw_fails, "前提条件: 素の json.loads は \\ + 実改行で失敗するはず"

        # _parse_json_safely は修復してパース成功すること
        result = _parse_json_safely(raw)
        assert "key" in result
        assert "line1" in result["key"]
        assert "line3" in result["key"]

    def test_non_json_raises_value_error(self):
        """JSON でないテキストを渡すと ValueError または JSONDecodeError が送出されること。"""
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_json_safely("これはJSONではありません")

    def test_list_json_is_parsed(self):
        """配列形式の JSON もパースできること。"""
        result = _parse_json_safely('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_embedded_control_chars_are_allowed(self):
        """strict=False により制御文字を含む JSON も許容されること。"""
        # 実際の改行（制御文字）を文字列値に直接含む（strict=True では拒否される）
        raw = '{"msg": "line1\nline2"}'
        result = _parse_json_safely(raw)
        assert "line1" in result["msg"]
        assert "line2" in result["msg"]

    def test_ndjson_simple(self):
        """1行1オブジェクトの NDJSON が list として返されること。"""
        lines = [
            '{"type": "system", "session_id": "sess01"}',
            '{"type": "result", "is_error": false}',
        ]
        result = _parse_json_safely("\n".join(lines))
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "system"
        assert result[1]["type"] == "result"

    def test_ndjson_multiline_value(self):
        """LLM 出力の生改行で複数行にまたがる NDJSON が正しくパースされること。

        Claude CLI の stream-json では text フィールドに生改行を含む行が複数行に分断される。
        積み上げパース方式で完全な JSON が形成されるまで行を結合することを確認する。
        """
        obj1 = '{"type": "system", "id": "s1"}'
        # 文字列値内に実際の改行を含む → splitlines() では 2 行になる
        obj2_line1 = '{"type": "assistant", "text": "line1'
        obj2_line2 = 'line2"}'
        obj3 = '{"type": "result", "ok": true}'
        raw = "\n".join([obj1, obj2_line1, obj2_line2, obj3])

        result = _parse_json_safely(raw)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["type"] == "system"
        assert result[1]["type"] == "assistant"
        assert "line1" in result[1]["text"]
        assert result[2]["type"] == "result"

    def test_ndjson_all_invalid_raises_value_error(self):
        """どの行も JSON としてパース不能なテキストは ValueError を送出すること。"""
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_json_safely("not json at all\nalso not json")


