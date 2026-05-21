"""WorkingMemory manager: SQLite と LanceStore を協調させてワーキングメモリを管理する。

# 設計方針

ワーキングメモリは「並行する複数の認知ストリーム」をスレッド方式で表現する層。

SQLite（working_memory_threads / working_memory_posts）がスレッド本体・ポストの source of truth、
LanceStore（working_memory_threads テーブル）が embedding 検索インデックス。
両者は同じ ``thread_id`` で紐付く。LanceStore の index には
``summary + 最新ポスト本文`` を結合したテキストを embed する。

# type 別の扱い

  - emotion / body : 各キャラに1本のみ。持続的な感情・身体状態のサマリ。固定注入。
  - relation       : 関係相手ごとに1本。固定注入（対話相手の分のみ）。
  - task / topic   : 無制限。解決を目指す。heat 上位 TopK で自動想起。

emotion/body/relation は本数が限られるため heat 計算の対象とせず、常に固定注入する。
task/topic のみ heat = importance × 時間減衰 × relevance でスコアリングする。
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from backend.repositories.lance.store import LanceStore
from backend.repositories.sqlite.store import SQLiteStore
from backend.services.memory.decay import (
    distance_to_similarity,
    elapsed_days_since,
    exp_decay,
)

logger = logging.getLogger(__name__)


class WorkingMemoryManager:
    """SQLite と LanceStore を協調させてワーキングメモリのスレッド・ポストを管理するクラス。

    Attributes:
        sqlite: SQLiteStore インスタンス。
        vector_store: LanceStore インスタンス（ベクトル検索層）。
    """

    # 有効なスレッド種別。
    VALID_TYPES = frozenset({"emotion", "body", "task", "topic", "relation"})
    # キャラごとに1本だけ存在できる種別。
    _SINGLETON_TYPES = frozenset({"emotion", "body"})
    # heat 想起の対象となる種別（解決を目指す種別のみ）。
    _RECALLABLE_TYPES = frozenset({"task", "topic"})
    # type 別の半減期（日）。task/topic のみ定義。relation/emotion/body は固定注入で対象外。
    _WM_HALF_LIFE = {"task": 14.0, "topic": 3.0}

    def __init__(self, sqlite: SQLiteStore, vector_store: LanceStore):
        """WorkingMemoryManager を初期化する。

        Args:
            sqlite: SQLite ストア。
            vector_store: ベクトルストア（LanceStore）。
        """
        self.sqlite = sqlite
        self.vector_store = vector_store

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------

    def _index_text(self, thread) -> str:
        """スレッドの embedding 素材（summary + 最新ポスト本文）を組み立てる。"""
        text = (thread.summary or "").strip()
        latest = self.sqlite.get_latest_working_memory_post(thread.id)
        if latest and latest.content:
            text = (text + "\n" + latest.content).strip()
        return text

    def _reindex_thread(self, thread_id: str) -> None:
        """スレッドの LanceStore embedding を最新状態で upsert する。

        summary 更新・ポスト追加・importance/is_open 変更のいずれでも呼ぶ。
        """
        thread = self.sqlite.get_working_memory_thread(thread_id)
        if not thread:
            return
        index_text = self._index_text(thread)
        if not index_text:
            return
        self.vector_store.upsert_working_memory_thread(
            thread_id=thread.id,
            index_text=index_text,
            character_id=thread.character_id,
            metadata={
                "type": thread.type,
                "importance": thread.importance,
                "is_open": thread.is_open,
            },
        )

    def _thread_to_dict(self, thread, include_posts: bool = False,
                         include_latest_post: bool = False) -> dict:
        """スレッド ORM を注入・API 用の dict に変換する。

        Args:
            thread: WorkingMemoryThread ORM オブジェクト。
            include_posts: True なら全ポストを ``posts`` キーに含める（open_working_memory_thread 用）。
            include_latest_post: True なら最新ポスト本文を ``latest_post`` キーに含める。
        """
        d = {
            "id": thread.id,
            "character_id": thread.character_id,
            "type": thread.type,
            "summary": thread.summary,
            "atmosphere_tag": thread.atmosphere_tag,
            "importance": thread.importance,
            "is_open": bool(thread.is_open),
            "relation_target": thread.relation_target,
            "created_at": thread.created_at.isoformat(timespec="seconds") if thread.created_at else None,
            "updated_at": thread.updated_at.isoformat(timespec="seconds") if thread.updated_at else None,
        }
        if include_latest_post or include_posts:
            latest = self.sqlite.get_latest_working_memory_post(thread.id)
            d["latest_post"] = latest.content if latest else None
        if include_posts:
            posts = self.sqlite.list_working_memory_posts(thread.id)
            d["posts"] = [
                {
                    "id": p.id,
                    "content": p.content,
                    "created_at": p.created_at.isoformat(timespec="seconds") if p.created_at else None,
                }
                for p in posts
            ]
        return d

    # ------------------------------------------------------------------
    # 作成・更新（能動ツール / Chronicle から呼ばれる）
    # ------------------------------------------------------------------

    def create_thread(
        self,
        character_id: str,
        type: str,
        summary: str,
        atmosphere_tag: str = "",
        importance: float = 0.5,
        relation_target: Optional[str] = None,
        content: Optional[str] = None,
    ) -> dict:
        """ワーキングメモリスレッドを新規作成する。

        type 別の本数制約をここで担保する:
          - emotion / body : 既に1本存在すれば作成を拒否（ValueError）。更新は thread_id 指定で行う。
          - relation       : relation_target 必須。同一相手のスレッドが既にあれば拒否。

        Args:
            character_id: キャラクター ID。
            type: スレッド種別（emotion/body/task/topic/relation）。
            summary: スレッドのタイトル相当。
            atmosphere_tag: 質感を表す短いタグ。
            importance: 重要度 0.0-1.0。
            relation_target: relation 型のときの相手識別子。
            content: 指定時は作成直後に最初のポストとして追加する。

        Returns:
            作成したスレッドの dict。

        Raises:
            ValueError: type 不正、または本数制約に違反した場合。
        """
        if type not in self.VALID_TYPES:
            raise ValueError(
                f"不正なスレッド種別 '{type}'。有効な種別: {sorted(self.VALID_TYPES)}"
            )

        if type in self._SINGLETON_TYPES:
            existing = self.sqlite.list_working_memory_threads(character_id, type=type)
            if existing:
                raise ValueError(
                    f"'{type}' スレッドは既に存在します（1本のみ）。"
                    f"更新は thread_id='{existing[0].id}' を指定して行ってください。"
                )

        if type == "relation":
            if not relation_target:
                raise ValueError("relation 型スレッドには relation_target が必須です。")
            existing = self.sqlite.get_working_memory_thread_by_relation(character_id, relation_target)
            if existing:
                raise ValueError(
                    f"'{relation_target}' との relation スレッドは既に存在します。"
                    f"更新は thread_id='{existing.id}' を指定して行ってください。"
                )

        importance = max(0.0, min(1.0, float(importance)))
        thread_id = str(uuid.uuid4())
        self.sqlite.add_working_memory_thread(
            thread_id=thread_id,
            character_id=character_id,
            type=type,
            summary=summary,
            atmosphere_tag=atmosphere_tag,
            importance=importance,
            relation_target=relation_target if type == "relation" else None,
        )
        if content:
            post_id = str(uuid.uuid4())
            self.sqlite.add_working_memory_post(post_id, thread_id, content)
        self._reindex_thread(thread_id)
        thread = self.sqlite.get_working_memory_thread(thread_id)
        logger.info(
            "WM thread 作成 char=%s type=%s id=%s summary=%.40s",
            character_id, type, thread_id, summary,
        )
        return self._thread_to_dict(thread, include_latest_post=True)

    def add_post(self, thread_id: str, content: str) -> dict:
        """既存スレッドにポストを追加する。embedding も更新する。

        Returns:
            更新後のスレッド dict（最新ポスト込み）。

        Raises:
            ValueError: スレッドが存在しない場合。
        """
        thread = self.sqlite.get_working_memory_thread(thread_id)
        if not thread:
            raise ValueError(f"スレッド '{thread_id}' が見つかりません。")
        post_id = str(uuid.uuid4())
        self.sqlite.add_working_memory_post(post_id, thread_id, content)
        self._reindex_thread(thread_id)
        thread = self.sqlite.get_working_memory_thread(thread_id)
        return self._thread_to_dict(thread, include_latest_post=True)

    def update_thread(
        self,
        thread_id: str,
        summary: Optional[str] = None,
        atmosphere_tag: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> dict:
        """スレッドの summary / atmosphere_tag / importance を部分更新する。

        Returns:
            更新後のスレッド dict。

        Raises:
            ValueError: スレッドが存在しない場合。
        """
        thread = self.sqlite.get_working_memory_thread(thread_id)
        if not thread:
            raise ValueError(f"スレッド '{thread_id}' が見つかりません。")
        if importance is not None:
            importance = max(0.0, min(1.0, float(importance)))
        self.sqlite.update_working_memory_thread(
            thread_id,
            summary=summary,
            atmosphere_tag=atmosphere_tag,
            importance=importance,
            touch=True,
        )
        # summary / importance が変わると embedding index・metadata に影響する
        self._reindex_thread(thread_id)
        thread = self.sqlite.get_working_memory_thread(thread_id)
        return self._thread_to_dict(thread, include_latest_post=True)

    def set_open(self, thread_id: str, is_open: bool) -> bool:
        """スレッドの is_open フラグを更新する（Chronicle 専用）。

        Returns:
            更新成否。
        """
        ok = self.sqlite.update_working_memory_thread(thread_id, is_open=is_open)
        if ok:
            self._reindex_thread(thread_id)
        return ok

    def delete_thread(self, thread_id: str) -> bool:
        """スレッドと配下ポストを物理削除する（SQLite → LanceStore）。"""
        ok = self.sqlite.delete_working_memory_thread(thread_id)
        if ok:
            self.vector_store.delete_working_memory_thread(thread_id)
        return ok

    # ------------------------------------------------------------------
    # 参照（システムプロンプト注入 / ツール）
    # ------------------------------------------------------------------

    def list_all_threads(self, character_id: str) -> list[dict]:
        """全スレッド（Open/Close 問わず）を dict リストで返す。

        self_history 代替の「全スレッド一覧」注入に使う。最新ポストは含めない。
        """
        threads = self.sqlite.list_working_memory_threads(character_id)
        return [self._thread_to_dict(t) for t in threads]

    def get_fixed_threads(
        self,
        character_id: str,
        participants: Optional[list[str]] = None,
    ) -> list[dict]:
        """固定注入対象（emotion / body / relation）のスレッドを返す。

        emotion / body は存在すれば1本ずつ。relation は対話相手に対応するもののみ。

        Args:
            character_id: キャラクター ID。
            participants: 現在の対話相手の識別子リスト。指定時は relation スレッドを
                relation_target がこのリストに含まれるものだけに絞る。None なら全 relation。

        Returns:
            最新ポスト込みのスレッド dict リスト。
        """
        result: list[dict] = []
        for t in self.sqlite.list_working_memory_threads(character_id, type="emotion"):
            result.append(self._thread_to_dict(t, include_latest_post=True))
        for t in self.sqlite.list_working_memory_threads(character_id, type="body"):
            result.append(self._thread_to_dict(t, include_latest_post=True))
        for t in self.sqlite.list_working_memory_threads(character_id, type="relation"):
            if participants is not None and t.relation_target not in participants:
                continue
            result.append(self._thread_to_dict(t, include_latest_post=True))
        return result

    def get_thread_detail(self, thread_id: str) -> Optional[dict]:
        """スレッド1件＋全ポストを dict で返す（open_working_memory_thread ツール用）。

        Returns:
            スレッド dict（``posts`` キーに全ポスト）。存在しなければ None。
        """
        thread = self.sqlite.get_working_memory_thread(thread_id)
        if not thread:
            return None
        return self._thread_to_dict(thread, include_posts=True)

    def list_threads_by_type(
        self,
        character_id: str,
        type: Optional[str] = None,
        is_open: Optional[bool] = None,
    ) -> list[dict]:
        """type / is_open で絞り込んだスレッド一覧を返す（Chronicle・UI 用）。"""
        threads = self.sqlite.list_working_memory_threads(character_id, type=type, is_open=is_open)
        return [self._thread_to_dict(t, include_latest_post=True) for t in threads]

    def recall_threads(
        self,
        character_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """task/topic の Open スレッドを heat 上位 TopK で想起する。

        heat = importance × decay(elapsed_days, type) × relevance(query)。
          - decay : type 別半減期（task=14日 / topic=3日）の指数減衰。
          - relevance : クエリとの cosine 類似度。
          - elapsed : last_touched_at（なければ created_at）からの経過日数。

        emotion/body/relation は固定注入のため対象外（get_fixed_threads を使うこと）。

        Args:
            character_id: キャラクター ID。
            query: 検索クエリ（直近のユーザー発言など）。
            top_k: 返す最大件数。

        Returns:
            heat 降順のスレッド dict リスト（``heat`` キー付き、最新ポスト込み）。
        """
        # heat 計算でリランクするため多めに取得する
        fetch_k = max(top_k * 2, top_k)
        results = self.vector_store.recall_working_memory_threads(
            query,
            character_id,
            top_k=fetch_k,
            where={"type": {"$in": ["task", "topic"]}, "is_open": 1},
        )
        now = datetime.now()
        scored: list[dict] = []
        for r in results:
            thread = self.sqlite.get_working_memory_thread(r.get("id", ""))
            if not thread or not thread.is_open:
                continue
            if thread.type not in self._RECALLABLE_TYPES:
                continue
            base_time = thread.last_touched_at or thread.created_at
            elapsed = elapsed_days_since(base_time, now)
            half_life = self._WM_HALF_LIFE.get(thread.type, 14.0)
            decay = exp_decay(1.0, elapsed, half_life)
            relevance = distance_to_similarity(r.get("distance", 1.0))
            heat = thread.importance * decay * relevance
            d = self._thread_to_dict(thread, include_latest_post=True)
            d["heat"] = heat
            scored.append(d)
        scored.sort(key=lambda x: x.get("heat", 0.0), reverse=True)
        return scored[:top_k]
