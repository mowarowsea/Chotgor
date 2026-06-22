"""recall 表示の origin ラベル付与のテスト（format.py）。

「卓を囲んだ友達が TRPG の記憶を覚えていないのは不自然」「うつつ由来の記憶を
通常チャットで誤って『ユーザと共有した現実』として語ると不自然」という立て付けから、
recall された記憶／スレッドの由来（origin）をキャラクター本人が表示時点で識別できる
よう、format_recalled_memories / format_recalled_threads はそれぞれ:

    - origin="interlude" → 行頭に "[TRPGでの記憶] "
    - origin="usual"     → 行頭に "[うつつでの記憶] "
    - origin="real" / None / 未知値 → ラベルなし（通常チャットと同一扱い）

を付与する。本テスト群はこの付与ルールを検証する。記憶整形は SSE 行単位パースに
載るため、改行 / 行頭 / 既存フォーマット（``[category]``・``⟦thread⟧`` マーカー）と
共存することも確認する。
"""

from backend.services.memory.format import (
    format_recalled_memories,
    format_recalled_threads,
)


# ---------------------------------------------------------------------------
# format_recalled_memories: 長期記憶の表示整形
# ---------------------------------------------------------------------------


class TestFormatRecalledMemoriesOriginLabel:
    """format_recalled_memories の origin ラベル付与の検証。

    metadata.origin 値に応じて行頭ラベル（TRPGでの記憶／うつつでの記憶／無し）が
    正しく差し込まれること、ラベルと既存フォーマット ``[category]`` が共存することを
    確認する。
    """

    def _make_mem(self, origin: str | None, content: str = "ナナと卓を囲んだ"):
        """origin 指定でテスト用の recall 結果 dict を作るヘルパー。"""
        meta: dict = {"category": "contextual"}
        if origin is not None:
            meta["origin"] = origin
        return {"content": content, "metadata": meta, "hybrid_score": 0.42}

    def test_interlude_origin_prefixed_with_trpg_label(self):
        """origin="interlude" の記憶は行頭に "[TRPGでの記憶] " が付くこと。"""
        text = format_recalled_memories([self._make_mem("interlude", content="ナナの作戦")])
        line = text.strip().splitlines()[0]
        assert line.startswith("[TRPGでの記憶] ")
        # 既存フォーマット（カテゴリ括弧と本文・score）も保たれる
        assert "[contextual]" in line
        assert "ナナの作戦" in line
        assert "(score: 0.42)" in line

    def test_usual_origin_prefixed_with_usual_label(self):
        """origin="usual" の記憶は行頭に "[うつつでの記憶] " が付くこと。"""
        text = format_recalled_memories([self._make_mem("usual", content="一人で散歩した")])
        line = text.strip().splitlines()[0]
        assert line.startswith("[うつつでの記憶] ")
        assert "[contextual]" in line
        assert "一人で散歩した" in line

    def test_real_origin_has_no_prefix(self):
        """origin="real" の記憶は行頭ラベルなし（既存通常チャットと同じ整形）。"""
        text = format_recalled_memories([self._make_mem("real", content="昨日の会話")])
        line = text.strip().splitlines()[0]
        assert not line.startswith("[TRPGでの記憶]")
        assert not line.startswith("[うつつでの記憶]")
        # 通常の `[category]` プレフィックスから始まる
        assert line.startswith("[contextual]")

    def test_missing_origin_metadata_treated_as_real(self):
        """origin が metadata に無い記憶（旧データ）は real 扱いで行頭ラベルが付かないこと。"""
        text = format_recalled_memories([self._make_mem(None, content="origin 欠落")])
        line = text.strip().splitlines()[0]
        assert line.startswith("[contextual]")

    def test_unknown_origin_treated_as_real(self):
        """未知の origin 値（誤値）は real 扱いにフォールバックすること（防御的挙動）。"""
        text = format_recalled_memories([self._make_mem("unknown_origin", content="未知の origin")])
        line = text.strip().splitlines()[0]
        assert line.startswith("[contextual]")

    def test_mixed_origins_each_get_appropriate_label(self):
        """複数記憶のリストで origin ごとに別のラベルが正しく付くこと。"""
        text = format_recalled_memories([
            self._make_mem("real", content="現実A"),
            self._make_mem("interlude", content="卓A"),
            self._make_mem("usual", content="日常A"),
        ])
        lines = text.strip().splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("[contextual]")
        assert lines[1].startswith("[TRPGでの記憶] ")
        assert lines[2].startswith("[うつつでの記憶] ")

    def test_empty_input_returns_empty_string(self):
        """空リストは空文字列を返すこと（後方互換）。"""
        assert format_recalled_memories([]) == ""


# ---------------------------------------------------------------------------
# format_recalled_threads: WM スレッドの表示整形
# ---------------------------------------------------------------------------


_THREAD_LINE_PREFIX = "⟦thread⟧"


class TestFormatRecalledThreadsOriginLabel:
    """format_recalled_threads の origin ラベル付与の検証。

    フロントエンドは行頭マーカー ``⟦thread⟧`` でスレッド行を識別するため、origin
    ラベルはマーカーの直後に挿入される。既存の `[type]`・〈atmosphere〉・→ latest 形式
    と共存することを確認する。
    """

    def _make_thread(self, origin: str | None, summary: str = "卓の余韻"):
        """origin 指定でテスト用のスレッド dict を作るヘルパー。"""
        t = {
            "type": "topic",
            "summary": summary,
            "atmosphere_tag": "緊張",
            "latest_post": "最後のポスト",
        }
        if origin is not None:
            t["origin"] = origin
        return t

    def test_interlude_thread_marker_then_trpg_label(self):
        """origin="interlude" のスレッドは「⟦thread⟧ [TRPGでの記憶] [type] ...」の順で並ぶこと。"""
        text = format_recalled_threads([self._make_thread("interlude", summary="卓で起きたこと")])
        line = text.strip().splitlines()[0]
        # ⟦thread⟧ マーカー → origin ラベル → 既存フォーマット
        assert line.startswith(f"{_THREAD_LINE_PREFIX} [TRPGでの記憶] [topic] ")
        assert "卓で起きたこと" in line
        assert "〈緊張〉" in line
        assert "→ 最後のポスト" in line

    def test_usual_thread_marker_then_usual_label(self):
        """origin="usual" のスレッドはマーカー直後に "[うつつでの記憶] " が付くこと。"""
        text = format_recalled_threads([self._make_thread("usual", summary="独りの夜")])
        line = text.strip().splitlines()[0]
        assert line.startswith(f"{_THREAD_LINE_PREFIX} [うつつでの記憶] [topic] ")

    def test_real_thread_has_no_origin_label(self):
        """origin="real" のスレッドはマーカー直後に origin ラベルが付かないこと。"""
        text = format_recalled_threads([self._make_thread("real", summary="ユーザと話したこと")])
        line = text.strip().splitlines()[0]
        # マーカー直後に [topic] が来る（origin ラベル無し）
        assert line.startswith(f"{_THREAD_LINE_PREFIX} [topic] ")

    def test_missing_origin_treated_as_real(self):
        """origin が dict に無いスレッド（旧データ）は real 扱いで origin ラベルなし。"""
        text = format_recalled_threads([self._make_thread(None, summary="origin 欠落")])
        line = text.strip().splitlines()[0]
        assert line.startswith(f"{_THREAD_LINE_PREFIX} [topic] ")

    def test_empty_input_returns_empty_string(self):
        """空リストは空文字列を返すこと（後方互換）。"""
        assert format_recalled_threads([]) == ""


# ---------------------------------------------------------------------------
# origin_label_prefix: 公開ヘルパの未知値ハンドリング・logger 連携
# ---------------------------------------------------------------------------


class TestOriginLabelPrefixHelper:
    """origin_label_prefix の公開 API としての挙動を検証する。

    `_origin_label` から rename した公開ヘルパ。format / executor / recaller / chronicle
    の各経路から共通利用されるため、未知値・None・空文字の扱いを契約として固定しておく。
    findings #12 (未知 origin 検出時の警告ログ) もここで確認する。
    """

    def test_real_returns_empty(self):
        """real は空文字を返す（ラベル無し）。"""
        from backend.services.memory.format import origin_label_prefix
        assert origin_label_prefix("real") == ""

    def test_usual_returns_usual_label(self):
        """usual は "[うつつでの記憶] " を返す。"""
        from backend.services.memory.format import origin_label_prefix
        assert origin_label_prefix("usual") == "[うつつでの記憶] "

    def test_interlude_returns_trpg_label(self):
        """interlude は "[TRPGでの記憶] " を返す。"""
        from backend.services.memory.format import origin_label_prefix
        assert origin_label_prefix("interlude") == "[TRPGでの記憶] "

    def test_none_returns_empty(self):
        """None は real 扱い（ラベル無し）。"""
        from backend.services.memory.format import origin_label_prefix
        assert origin_label_prefix(None) == ""

    def test_empty_string_returns_empty(self):
        """空文字も real 扱い（ラベル無し）。"""
        from backend.services.memory.format import origin_label_prefix
        assert origin_label_prefix("") == ""

    def test_unknown_value_logs_warning_and_falls_back(self, caplog):
        """未知 origin 値はラベル無しに倒れつつ logger.warning が記録されること（findings #12）。"""
        import logging
        from backend.services.memory.format import origin_label_prefix
        with caplog.at_level(logging.WARNING, logger="backend.services.memory.format"):
            result = origin_label_prefix("unknown_origin_value")
        assert result == ""
        assert any(
            "未知の origin 値" in rec.message and "unknown_origin_value" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# power_recall 経路の origin 注入
# ---------------------------------------------------------------------------


class TestPowerRecallOriginInjection:
    """manager.power_recall が SQLite から origin を引いて metadata に埋め込むことの検証
    （findings #4）。

    旧実装は vector_store.recall_inscribed_memory を直接呼ぶだけで origin が落ちていた。
    新実装では各記憶ヒットに対し SQLite ORM の origin を読み取って metadata.origin に
    格納する。
    """

    def test_power_recall_populates_metadata_origin_from_sqlite(self, sqlite_store):
        """vector_store ヒットに origin が無くても、SQLite ORM から引いて埋め込まれること。"""
        import uuid
        from unittest.mock import MagicMock
        from backend.services.memory.manager import InscribedMemoryManager

        char_id = str(uuid.uuid4())
        sqlite_store.create_character(char_id, "テスト")

        # SQLite に origin="interlude" の記憶を 1 件作る
        mem_id = str(uuid.uuid4())
        sqlite_store.create_inscribed_memory(
            mem_id, char_id, "卓で起きた一幕", "contextual",
            contextual_importance=0.7, semantic_importance=0.0,
            identity_importance=0.0, user_importance=0.0,
            origin="interlude",
        )

        # vector_store は origin 無しで返してくるモック
        mock_vs = MagicMock()
        mock_vs.recall_inscribed_memory.return_value = [
            {"id": mem_id, "content": "卓で起きた一幕", "distance": 0.1, "metadata": {"category": "contextual"}},
        ]
        mock_vs.recall_chat_turns.return_value = []

        mgr = InscribedMemoryManager(sqlite_store, mock_vs)
        result = mgr.power_recall(character_id=char_id, query="卓", top_k=5)

        memories = result["inscribed_memories"]
        assert len(memories) == 1
        # SQLite から引いた origin が metadata に注入されている
        assert memories[0]["metadata"]["origin"] == "interlude"
