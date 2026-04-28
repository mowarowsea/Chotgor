"""Memory manager: coordinates SQLite and ChromaDB, handles write/recall/cleanup."""

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from backend.repositories.chroma.store import ChromaStore
from backend.repositories.sqlite.store import SQLiteStore

logger = logging.getLogger(__name__)

# ChromaDB書き込みリトライ設定
_CHROMA_MAX_RETRIES = 5
_CHROMA_RETRY_INTERVAL_SEC = 30.0   # リトライ間隔（秒）
_CHROMA_RETRY_POLL_SEC = 5.0        # ワーカーポーリング間隔（秒）


@dataclass
class _PendingChromaWrite:
    """バックグラウンドリトライ待ちのChromaDB書き込みタスク。

    Attributes:
        character_id: 書き込み対象キャラクターID。_chroma_failed_chars への登録に使用。
        fn_name: 実行する関数名。"add_memory" / "delete_memory" のいずれか。
        kwargs: 関数に渡すキーワード引数。
        attempts: これまでの試行回数（初回試行は _schedule_chroma_write で行うため、
                  キューに積まれた時点で 1 以上）。
        next_retry_after: 次にリトライを試みる monotonic 時刻。
    """

    character_id: str
    fn_name: Literal["add_memory", "delete_memory"]
    kwargs: dict
    attempts: int = 0
    next_retry_after: float = field(default_factory=time.monotonic)


# recall_memory に挿入する警告エントリ。SQLiteに存在しないIDを使うため remember() は安全にスキップする。
_CHROMA_SYNC_WARNING_MEMORY: dict = {
    "id": "_chroma_sync_error",
    "content": (
        "⚠️ システム通知: 記憶インデックス（ChromaDB）への書き込みが繰り返し失敗しています。"
        "Settings → Reindex を実行して記憶を再インデックスしてください。"
    ),
    "distance": 0.0,
    "metadata": {"category": "_system"},
    "hybrid_score": 999.0,
    "decayed_score": 0.0,
    "semantic_similarity": 0.0,
}


class MemoryManager:
    """SQLiteとChromaDBを協調させて記憶の書き込み・想起・管理を行うクラス。

    設計方針: SQLite = source of truth、ChromaDB = ベストエフォートの検索インデックス。
    ChromaDB書き込みはSQLiteコミット後に行い、失敗時はバックグラウンドでリトライする。
    最大5回失敗した場合はエラーログを記録し、次回のrecall_memory/power_recall結果に
    再インデックスを促す警告を挿入する。

    Attributes:
        sqlite: SQLiteStoreインスタンス。
        chroma: ChromaStoreインスタンス。
        _chroma_failed_chars: 最大リトライを超えて失敗したキャラのIDセット（in-memory）。
        _pending_writes: バックグラウンドリトライ待ちのタスクリスト。
        _pending_lock: _pending_writes へのスレッドセーフなアクセスを保護するロック。
        _stop_event: graceful shutdown用の停止フラグ。
        _retry_thread: バックグラウンドリトライワーカースレッド。
    """

    def __init__(self, sqlite: SQLiteStore, chroma: ChromaStore):
        """MemoryManagerを初期化し、バックグラウンドリトライワーカーを起動する。"""
        self.sqlite = sqlite
        self.chroma = chroma
        self._chroma_failed_chars: set[str] = set()
        self._pending_writes: list[_PendingChromaWrite] = []
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._retry_thread = threading.Thread(
            target=self._chroma_retry_worker, daemon=True, name="chroma-retry"
        )
        self._retry_thread.start()

    def _execute_chroma_write(self, task: _PendingChromaWrite) -> None:
        """タスクのChromaDB書き込みを実行する。失敗時は例外を伝播させる。

        character_id はタスクの tracking フィールドから自動注入するため、
        kwargs には含めなくてよい。

        Args:
            task: 実行するChromaDB書き込みタスク。
        """
        if task.fn_name == "add_memory":
            self.chroma.add_memory(character_id=task.character_id, **task.kwargs)
        elif task.fn_name == "delete_memory":
            self.chroma.delete_memory(character_id=task.character_id, **task.kwargs)
        else:
            raise ValueError(f"未知の ChromaDB 書き込み操作: {task.fn_name!r}")

    def _process_pending_writes(self) -> None:
        """リトライキューを1回処理する。now以前のタスクのみ実行する。

        バックグラウンドスレッドから定期的に呼ばれるほか、テストから直接呼び出せる。
        """
        now = time.monotonic()
        with self._pending_lock:
            ready = [t for t in self._pending_writes if now >= t.next_retry_after]

        for task in ready:
            try:
                self._execute_chroma_write(task)
                with self._pending_lock:
                    if task in self._pending_writes:
                        self._pending_writes.remove(task)
                logger.info(
                    "ChromaDB リトライ成功 fn=%s char=%s attempt=%d",
                    task.fn_name, task.character_id, task.attempts,
                )
            except Exception as e:
                task.attempts += 1
                if task.attempts >= _CHROMA_MAX_RETRIES:
                    logger.error(
                        "ChromaDB 書き込み最終失敗（%d回試行） fn=%s char=%s error=%s",
                        task.attempts, task.fn_name, task.character_id, e,
                    )
                    self._chroma_failed_chars.add(task.character_id)
                    with self._pending_lock:
                        if task in self._pending_writes:
                            self._pending_writes.remove(task)
                else:
                    task.next_retry_after = time.monotonic() + _CHROMA_RETRY_INTERVAL_SEC
                    logger.warning(
                        "ChromaDB 書き込み失敗（%d回目） fn=%s char=%s error=%s → %gs後にリトライ",
                        task.attempts, task.fn_name, task.character_id, e,
                        _CHROMA_RETRY_INTERVAL_SEC,
                    )

    def _chroma_retry_worker(self) -> None:
        """ChromaDB書き込みをバックグラウンドでリトライするワーカースレッド。

        _CHROMA_RETRY_POLL_SEC ごとに _process_pending_writes() を呼び出す。
        stop() が呼ばれると _stop_event がセットされ、次のポーリング前に終了する。
        """
        while not self._stop_event.wait(_CHROMA_RETRY_POLL_SEC):
            self._process_pending_writes()

    def stop(self) -> None:
        """リトライスレッドを graceful に停止する。lifespan shutdown 時に呼ぶ。"""
        self._stop_event.set()
        self._retry_thread.join(timeout=10)

    def _schedule_chroma_write(self, fn_name: str, character_id: str, **kwargs) -> None:
        """ChromaDB書き込みをスケジュールする。

        まず即時1回試行する。成功すればそのまま終了し、失敗した場合はリトライキューに追加する。
        これによりSQLiteコミット後のChromaDB書き込み失敗がユーザー体験に影響しない。

        Args:
            fn_name: 実行する関数名（"add_memory" / "delete_memory"）。
            character_id: 書き込み対象のキャラクターID。
            **kwargs: 関数に渡すキーワード引数。
        """
        task = _PendingChromaWrite(character_id=character_id, fn_name=fn_name, kwargs=kwargs)
        try:
            self._execute_chroma_write(task)
        except Exception as e:
            logger.warning(
                "ChromaDB 書き込み初回失敗（リトライスケジュール） fn=%s char=%s error=%s",
                fn_name, character_id, e,
            )
            task.attempts = 1
            task.next_retry_after = time.monotonic() + _CHROMA_RETRY_INTERVAL_SEC
            with self._pending_lock:
                self._pending_writes.append(task)

    # 現役カテゴリ。これ以外は contextual と同じ速度で減衰させる
    _KNOWN_CATEGORIES = frozenset({"contextual", "semantic", "identity", "user"})

    def calculate_decayed_score(self, memory, now: Optional[datetime] = None) -> float:
        """Calculate the time-decayed importance score for a memory.

        Importance logic:
        - contextual: High weight (1.0), Fast decay (half-life ~4 days)
        - user: Medium weight (0.8), Medium decay (half-life ~10 days)
        - semantic: Medium weight (0.6), Slow decay (half-life ~20 days)
        - identity: Low weight (0.3), Very slow decay (half-life ~90 days)
        """
        if now is None:
            now = datetime.now()

        base_time = memory.last_accessed_at or memory.created_at
        hours_passed = (now - base_time).total_seconds() / 3600.0
        days_passed = hours_passed / 24.0
        if days_passed < 0:
            days_passed = 0.0

        # Math: e^(-lambda * t) where lambda = ln(2) / half_life
        ln2 = 0.69314718

        decay_contextual = memory.contextual_importance * math.exp(-(ln2 / 4.0) * days_passed)
        decay_user = memory.user_importance * math.exp(-(ln2 / 10.0) * days_passed)
        decay_semantic = memory.semantic_importance * math.exp(-(ln2 / 20.0) * days_passed)
        decay_identity = memory.identity_importance * math.exp(-(ln2 / 90.0) * days_passed)

        # Weighted sum
        score = (
            (decay_contextual * 1.0) +
            (decay_user * 0.8) +
            (decay_semantic * 0.6) +
            (decay_identity * 0.3)
        )
        return score

    def get_forgotten_candidates(
        self, character_id: str, threshold: float = 0.3, limit: int = 50
    ) -> list:
        """Return memories whose decayed score falls below the threshold."""
        now = datetime.now()
        candidates = []

        memories = self.sqlite.get_all_active_memories(character_id)

        for m in memories:
            score = self.calculate_decayed_score(m, now)
            if score < threshold:
                setattr(m, '_decayed_score', score)
                candidates.append(m)

        # Sort by score ascending (lowest score first)
        candidates.sort(key=lambda x: getattr(x, '_decayed_score', 0))
        return candidates[:limit]

    def write_memory(
        self,
        character_id: str,
        content: str,
        category: str = "general",
        contextual_importance: float = 0.5,
        semantic_importance: float = 0.5,
        identity_importance: float = 0.5,
        user_importance: float = 0.5,
        source_preset_id: Optional[str] = None,
    ) -> str:
        """記憶をSQLiteとChromaDBに書き込む。類似記憶があれば更新、なければ新規作成する。

        SQLiteへの書き込みを先に確定させ、ChromaDB書き込みはその後ベストエフォートで行う。
        ChromaDB書き込みが失敗した場合はバックグラウンドでリトライする。

        同一キャラクター・カテゴリ内でコサイン距離 < 0.2 の記憶が既存する場合は
        新規作成せずに上書き更新する（重複排除）。
        更新時は access_count と created_at を引き継ぎ、content・各importance を上書きする。

        Returns:
            書き込んだ記憶のmemory_id（更新の場合は既存ID、新規の場合は新規UUID）。
        """
        # カテゴリ別の更新判定閾値
        # identity は自己定義に関わる記憶のため、ほぼ同文（距離 < 0.05）のみ上書きする
        similarity_threshold = 0.05 if category == "identity" else 0.15

        # 同一カテゴリ内で類似記憶を検索する
        existing_id = self.chroma.find_similar_in_category(
            content=content,
            character_id=character_id,
            category=category,
            threshold=similarity_threshold,
        )

        if existing_id:
            # 類似記憶が見つかった → SQLiteをsoft-delete後、ChromaDBから削除する（ベストエフォート）
            self.sqlite.soft_delete_memory(existing_id)
            self._schedule_chroma_write(
                "delete_memory", character_id,
                memory_id=existing_id,
            )

        # 新規作成: まずSQLiteにコミットしてからChromaDBへ書き込む
        memory_id = str(uuid.uuid4())

        self.sqlite.create_memory(
            memory_id=memory_id,
            character_id=character_id,
            content=content,
            memory_category=category,
            contextual_importance=contextual_importance,
            semantic_importance=semantic_importance,
            identity_importance=identity_importance,
            user_importance=user_importance,
            source_preset_id=source_preset_id,
        )

        # SQLiteコミット完了後にChromaDBへ書き込む（失敗時はリトライキューへ）
        self._schedule_chroma_write(
            "add_memory", character_id,
            memory_id=memory_id,
            content=content,
            metadata={
                "category": category,
                "contextual_importance": contextual_importance,
                "semantic_importance": semantic_importance,
                "identity_importance": identity_importance,
                "user_importance": user_importance,
            },
        )

        return memory_id

    def recall_memory(
        self,
        character_id: str,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """類似度検索＋時間減衰リランクで記憶を想起する。

        処理フロー:
        1. ChromaDB から top_k * 2 件取得（セマンティック検索）
        2. SQLite で各記憶の時間減衰スコアを計算
        3. ハイブリッドスコアでリランクして top_k 件を返す
        4. ChromaDB同期失敗中の場合、先頭に再インデックスを促す警告を挿入する

        Args:
            character_id: キャラクターID。
            query: 検索クエリテキスト。
            top_k: 返す最大件数。
            where: ChromaDB の where フィルタ。recall_with_identity から使用。
        """
        # Fetch more candidates for reranking
        fetch_k = top_k * 2
        results = self.chroma.recall_memory(query, character_id, fetch_k, where=where)

        now = datetime.now()

        # Rerank and inject metadata
        reranked = []
        for mem in results:
            mem_id = mem.get("id")
            if not mem_id:
                continue

            try:
                m = self.sqlite.get_memory(mem_id)
                if not m:
                    logger.debug("recall_memory: ChromaDB にあるが SQLite に存在しない記憶をスキップ id=%s char=%s", mem_id, character_id)
                    continue
                # soft-delete 済み記憶はスキップ（ChromaDB から完全削除される前の過渡状態）
                if m.deleted_at is not None:
                    logger.debug("recall_memory: soft-delete 済み記憶をスキップ id=%s char=%s", mem_id, character_id)
                    continue

                # Calculate True Decayed Score
                decayed_score = self.calculate_decayed_score(m, now)

                # ChromaDB distance is lower-is-better (cosine distance: 0=identical, 2=opposite)
                # Convert distance to similarity (higher is better)
                semantic_similarity = max(0.0, 1.0 - (mem.get("distance", 1.0) / 2.0))

                # Hybrid score: Semantic Similarity + Decayed Score
                hybrid_score = (semantic_similarity * 0.5) + (decayed_score * 0.5)

                mem["hybrid_score"] = hybrid_score
                mem["decayed_score"] = decayed_score
                mem["semantic_similarity"] = semantic_similarity

                if m.created_at:
                    mem.setdefault("metadata", {})["created_at"] = m.created_at.isoformat(timespec="seconds")

                reranked.append(mem)
            except Exception:
                pass

        # Sort by hybrid_score descending
        reranked.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        final_results = reranked[:top_k]

        # ChromaDB同期失敗中の場合、先頭に警告エントリを挿入する
        # recall()はSQLiteに存在しないIDを安全にスキップするため警告エントリは問題ない
        if character_id in self._chroma_failed_chars:
            final_results.insert(0, _CHROMA_SYNC_WARNING_MEMORY.copy())

        # 想起回数をインクリメント（last_accessed_at は更新しない — decay タイマー保持のため）
        for mem in final_results:
            mem_id = mem.get("id")
            if mem_id:
                self.sqlite.remember(mem_id)

        return final_results

    def recall_with_identity(
        self,
        character_id: str,
        query: str,
        identity_top_k: int = 5,
        other_top_k: int = 10,
    ) -> tuple[list[dict], list[dict]]:
        """identity カテゴリとそれ以外を別枠で想起して返す。

        identity 記憶はキャラクターの自己定義に関わるため、スコアに関係なく常時注入する。
        各枠は独立して recall_memory を呼び出し、リランク済みの結果をそのまま返す。

        Args:
            character_id: キャラクターID。
            query: 検索クエリテキスト。
            identity_top_k: identity カテゴリから返す最大件数。
            other_top_k: identity 以外のカテゴリから返す最大件数。

        Returns:
            (identity_memories, other_memories) のタプル。
        """
        identity_memories = self.recall_memory(
            character_id=character_id,
            query=query,
            top_k=identity_top_k,
            where={"category": "identity"},
        )
        logger.info(
            "recall_with_identity: identity=%d char=%s query=%.40s",
            len(identity_memories), character_id, query,
        )

        other_memories = self.recall_memory(
            character_id=character_id,
            query=query,
            top_k=other_top_k,
            where={"category": {"$ne": "identity"}},
        )
        logger.info(
            "recall_with_identity: other=%d char=%s",
            len(other_memories), character_id,
        )

        return identity_memories, other_memories

    def power_recall(
        self,
        character_id: str,
        query: str,
        top_k: int = 10,
    ) -> dict[str, list[dict]]:
        """記憶コレクションとチャット履歴コレクションを横断して検索する（PowerRecall用）。

        通常のrecall_memoryと異なり、以下の点が異なる:
          - 時間減衰スコアによるリランクを行わない（意味的類似度のみで返す）
          - 論理削除レコードも除外しない（ChromaDB側での管理のみ）
          - 記憶コレクション（char_*）とチャット履歴コレクション（chat_*）を両方検索する
          - ChromaDB同期失敗中の場合、"_warning" キーに再インデックスを促すメッセージを含む

        Args:
            character_id: キャラクターID。
            query: 検索クエリテキスト。
            top_k: 各コレクションから取得する最大件数。

        Returns:
            {"memories": [...], "chat_turns": [...]} の辞書。
            ChromaDB同期失敗中は "_warning" キーも含む。
            各リストは id / content / distance / metadata を持つdict。
        """
        memories = self.chroma.recall_memory(query, character_id, top_k=top_k)
        chat_turns = self.chroma.recall_chat_turns(query, character_id, top_k=top_k)

        # 各ヒットに前後コンテキストを付与する。
        # 同一セッションへの重複クエリを避けるため、セッションIDごとに一度だけ取得する。
        session_messages: dict[str, list] = {}
        for turn in chat_turns:
            session_id = turn.get("metadata", {}).get("session_id")
            if not session_id:
                logger.warning("session_id なしの chat_turn をスキップ id=%s", turn.get("id"))
                turn["context"] = []
                continue
            if session_id not in session_messages:
                session_messages[session_id] = self.sqlite.list_chat_messages(session_id)
            turn["context"] = self._build_context_window(
                session_messages[session_id], turn["id"], window=2
            )

        logger.info(
            "完了 char=%s query=%.50s memories=%d chat_turns=%d",
            character_id, query, len(memories), len(chat_turns),
        )

        result: dict = {"memories": memories, "chat_turns": chat_turns}
        # ChromaDB同期失敗中はexecutor側で警告テキストを先頭に表示するためキーを追加する
        if character_id in self._chroma_failed_chars:
            result["_warning"] = (
                "⚠️ 記憶インデックス（ChromaDB）への書き込みが繰り返し失敗しています。"
                "Settings → Reindex を実行して再インデックスしてください。"
            )
        return result

    @staticmethod
    def _build_context_window(msgs: list, message_id: str, window: int) -> list[dict]:
        """セッションのメッセージリストから指定IDを中心にウィンドウを切り出す。

        Args:
            msgs: list_chat_messages() が返す ORM オブジェクトのリスト（時系列順）。
            message_id: 中心メッセージの ID。
            window: 前後に含めるメッセージ数。

        Returns:
            id / role / speaker_name / content / created_at / is_hit を持つ dict のリスト。
        """
        clean = [m for m in msgs if not getattr(m, "is_system_message", None)]
        ids = [m.id for m in clean]
        try:
            idx = ids.index(message_id)
        except ValueError:
            return []

        start = max(0, idx - window)
        end = min(len(clean), idx + window + 1)
        result = []
        for m in clean[start:end]:
            speaker = getattr(m, "character_name", None) or ("ユーザ" if m.role == "user" else "キャラクター")
            result.append(
                {
                    "id": m.id,
                    "role": m.role,
                    "speaker_name": speaker,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(timespec="seconds") if m.created_at else "",
                    "is_hit": m.id == message_id,
                }
            )
        return result

    def delete_memory(self, memory_id: str, character_id: str) -> bool:
        """SQLite ソフトデリート + ChromaDB ハードデリートを行う。

        SQLiteへの削除を先に確定させ、ChromaDB削除はその後ベストエフォートで行う。
        ChromaDB削除が失敗した場合はバックグラウンドでリトライする。

        Args:
            memory_id: 削除対象の記憶ID。
            character_id: 所属キャラクターID（ChromaDB の名前空間に使用）。

        Returns:
            SQLite の削除成否。False の場合は記憶が存在しないか既に削除済み。
        """
        ok = self.sqlite.soft_delete_memory(memory_id)
        if ok:
            # SQLiteコミット完了後にChromaDBから削除する（失敗時はリトライキューへ）
            self._schedule_chroma_write(
                "delete_memory", character_id,
                memory_id=memory_id,
            )
        return ok

    def restore_memory(self, memory_id: str) -> bool:
        """ソフトデリート済み記憶を復元する。

        ChromaDB は復元対象外（再インデックスは write_memory を使用）。
        SQLite の deleted_at を NULL に戻すだけ。

        Args:
            memory_id: 復元対象の記憶ID。

        Returns:
            復元成否。False の場合は記憶が存在しないか削除されていない。
        """
        return self.sqlite.restore_memory(memory_id)

    def delete_character_with_memories(self, character_id: str) -> bool:
        """キャラクターと全記憶を削除する。

        SQLiteのキャラクターレコードを先に削除してから、ChromaDBのコレクションを削除する。
        SQLite = source of truth のため、SQLiteの削除を先に確定させる。
        ChromaDB削除は失敗してもSQLiteのロールバックは行わない（ベストエフォート）。

        api/characters.py と api/ui.py の2箇所から共通利用するためにここで一元管理する。

        Args:
            character_id: 削除対象のキャラクターID。

        Returns:
            SQLite の削除成否。False の場合はキャラクターが存在しない。
        """
        result = self.sqlite.delete_character(character_id)
        if result:
            # SQLiteの削除が確定してからChromaDBのコレクションを削除する（ベストエフォート）
            self.chroma.delete_all_memories(character_id)
        return result

    def list_memories(
        self,
        character_id: str,
        category: Optional[str] = None,
        include_deleted: bool = False,
        sort_by: str = "created_at",
    ) -> list[dict]:
        """キャラクターの記憶一覧をdictリストで返す。"""
        mems = self.sqlite.list_memories(character_id, category, include_deleted, sort_by)
        return [
            {
                "id": m.id,
                "character_id": m.character_id,
                "content": m.content,
                "category": m.memory_category,
                "contextual_importance": m.contextual_importance,
                "semantic_importance": m.semantic_importance,
                "identity_importance": m.identity_importance,
                "user_importance": m.user_importance,
                "access_count": m.access_count,
                "last_accessed_at": (
                    m.last_accessed_at.isoformat() if m.last_accessed_at else None
                ),
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                "deleted_at": m.deleted_at.isoformat() if m.deleted_at else None,
                "source_preset_id": m.source_preset_id,
            }
            for m in mems
        ]
