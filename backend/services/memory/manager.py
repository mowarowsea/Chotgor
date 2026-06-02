"""InscribedMemoryManager: SQLite と LanceStore を協調させて保存記憶の書き込み・想起・管理を行う。

SQLite は記憶のメタデータ（カテゴリ・importance・タイムスタンプ等）を保持する
source of truth、LanceStore はベクトル + 検索インデックス。
両者は同じ ``memory_id`` で紐付き、書き込みは SQLite → LanceStore の順に行う。
"""

import logging
import uuid
from datetime import datetime
from backend.repositories.lance.store import LanceStore
from backend.repositories.sqlite.store import SQLiteStore
from backend.services.memory.decay import (
    distance_to_similarity,
    elapsed_days_since,
    exp_decay,
)

logger = logging.getLogger(__name__)


class InscribedMemoryManager:
    """SQLite と LanceStore を協調させて記憶の書き込み・想起・管理を行うクラス。

    Attributes:
        sqlite: SQLiteStore インスタンス。
        vector_store: LanceStore インスタンス（ベクトル検索層）。
    """

    def __init__(self, sqlite: SQLiteStore, vector_store: LanceStore):
        """InscribedMemoryManager を初期化する。

        Args:
            sqlite: SQLite ストア。
            vector_store: ベクトルストア（LanceStore）。
        """
        self.sqlite = sqlite
        self.vector_store = vector_store

    # 現役カテゴリ。これ以外は contextual と同じ速度で減衰させる
    _KNOWN_CATEGORIES = frozenset({"contextual", "semantic", "identity", "user"})

    def calculate_decayed_score(self, memory, now: datetime | None = None) -> float:
        """記憶の時間減衰込み importance score を計算する。

        Importance ロジック:
          - contextual: High weight (1.0), Fast decay (half-life ~4 days)
          - user:       Medium weight (0.8), Medium decay (half-life ~10 days)
          - semantic:   Medium weight (0.6), Slow decay (half-life ~20 days)
          - identity:   Low weight (0.3), Very slow decay (half-life ~90 days)
        """
        if now is None:
            now = datetime.now()

        base_time = memory.last_accessed_at or memory.created_at
        days_passed = elapsed_days_since(base_time, now)

        # 指数減衰 e^(-λt)（λ = ln2 / half_life）は decay.exp_decay に共通化済み
        decay_contextual = exp_decay(memory.contextual_importance, days_passed, 4.0)
        decay_user = exp_decay(memory.user_importance, days_passed, 10.0)
        decay_semantic = exp_decay(memory.semantic_importance, days_passed, 20.0)
        decay_identity = exp_decay(memory.identity_importance, days_passed, 90.0)

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
        """減衰スコアが threshold を下回ったアクティブ記憶を返す（忘却バッチ用）。"""
        now = datetime.now()
        candidates = []

        memories = self.sqlite.get_all_active_inscribed_memories(character_id)

        for m in memories:
            score = self.calculate_decayed_score(m, now)
            if score < threshold:
                setattr(m, '_decayed_score', score)
                candidates.append(m)

        # スコア昇順でソートする（最低スコア優先）
        candidates.sort(key=lambda x: getattr(x, '_decayed_score', 0))
        return candidates[:limit]

    def get_top_memorable(self, character_id: str, limit: int = 30) -> list:
        """減衰スコア降順で上位 limit 件のアクティブ記憶を返す（chronicle 用）。

        get_forgotten_candidates と対称で、「印象的な記憶」をセマンティック検索なしで取得する。
        各記憶に _decayed_score 属性を付与する。

        Args:
            character_id: キャラクター ID。
            limit: 返す最大件数。

        Returns:
            _decayed_score 属性付きの記憶 ORM オブジェクトのリスト（降順）。
        """
        now = datetime.now()
        memories = self.sqlite.get_all_active_inscribed_memories(character_id)
        for m in memories:
            setattr(m, '_decayed_score', self.calculate_decayed_score(m, now))
        memories.sort(key=lambda x: getattr(x, '_decayed_score', 0.0), reverse=True)
        return memories[:limit]

    def write_inscribed_memory(
        self,
        character_id: str,
        content: str,
        category: str = "general",
        contextual_importance: float = 0.5,
        semantic_importance: float = 0.5,
        identity_importance: float = 0.5,
        user_importance: float = 0.5,
        source_preset_id: str | None = None,
        force_insert: bool = False,
    ) -> str:
        """記憶を SQLite と LanceStore に書き込む。類似記憶があれば in-place 更新、なければ新規作成。

        SQLite を先にコミットしてから LanceStore に書き込む。LanceStore 書き込みは
        原子的で、失敗（embedding API 落ち等）は例外として呼び出し側に伝播する。

        同一キャラクター・カテゴリ内で類似既存記憶が見つかった場合は、
        既存 memory_id を再利用して in-place 上書きする（重複排除）。
        access_count / created_at / last_accessed_at は維持され、
        content / category / importances / updated_at のみ上書きされる。
        LanceStore 側は同一 ID で ``add_inscribed_memory`` を呼ぶことで内部 merge_insert により
        embedding と metadata が原子的に置き換わる（delete は不要）。

        Args:
            force_insert: True の場合、類似既存記憶があっても上書きせず必ず新規 UUID で挿入する。
                forget 蒸留バッチ専用フラグ。蒸留結果が「削除対象として提示した候補記憶」の
                ID に上書きされてしまうと、続く全件削除フェーズで蒸留物まで消えてしまうため。

        Returns:
            書き込んだ記憶の memory_id（更新の場合は既存ID、新規の場合は新規UUID）。
        """
        if force_insert:
            # forget 蒸留バッチ用パス: 類似検索をスキップして必ず新規 ID で挿入する。
            existing_id = None
            logger.info(
                "write_inscribed_memory: force_insert=True により類似検索スキップ char=%s category=%s",
                character_id, category,
            )
        else:
            # カテゴリ別の更新判定閾値
            # identity は自己定義に関わる記憶のため、ほぼ同文（距離 < 0.05）のみ上書きする
            similarity_threshold = 0.05 if category == "identity" else 0.15

            # 同一カテゴリ内で類似記憶を検索する
            existing_id = self.vector_store.find_similar_in_category(
                content=content,
                character_id=character_id,
                category=category,
                threshold=similarity_threshold,
            )

        if existing_id:
            logger.info(
                "write_inscribed_memory: 類似既存記憶あり → in-place 上書き char=%s category=%s existing_id=%s",
                character_id, category, existing_id,
            )
            # 既存 ID を再利用して in-place 更新する。SQLite 側を先に確定させる。
            updated = self.sqlite.update_inscribed_memory_for_overwrite(
                memory_id=existing_id,
                content=content,
                memory_category=category,
                contextual_importance=contextual_importance,
                semantic_importance=semantic_importance,
                identity_importance=identity_importance,
                user_importance=user_importance,
            )
            if updated:
                memory_id = existing_id
            else:
                # 既存IDが SQLite 側で見つからない／soft-delete 済み等の不整合状態。
                # ベクトル DB 側にだけ存在するゴーストレコードの可能性が高いので、
                # 新規 UUID で作り直す方向にフォールバックする。
                logger.warning(
                    "in-place 更新失敗（SQLite 側に existing_id=%s が見つからず）→ 新規作成にフォールバック",
                    existing_id,
                )
                existing_id = None

        if not existing_id:
            # 新規作成: まず SQLite にコミットしてから LanceStore へ書き込む
            memory_id = str(uuid.uuid4())
            logger.info(
                "write_inscribed_memory: 新規 UUID 作成 char=%s category=%s memory_id=%s force_insert=%s",
                character_id, category, memory_id, force_insert,
            )
            self.sqlite.create_inscribed_memory(
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

        # SQLite コミット完了後に LanceStore へ書き込む。
        # 同一 ID の場合は merge_insert により embedding と metadata が原子的に置き換わる。
        self.vector_store.add_inscribed_memory(
            memory_id=memory_id,
            content=content,
            character_id=character_id,
            metadata={
                "category": category,
                "contextual_importance": contextual_importance,
                "semantic_importance": semantic_importance,
                "identity_importance": identity_importance,
                "user_importance": user_importance,
            },
        )

        return memory_id

    def recall_inscribed_memory(
        self,
        character_id: str,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """類似度検索＋時間減衰リランクで記憶を想起する。

        処理フロー:
          1. LanceStore から top_k * 2 件取得（セマンティック検索）
          2. SQLite で各記憶の時間減衰スコアを計算
          3. ハイブリッドスコアでリランクして top_k 件を返す

        Args:
            character_id: キャラクター ID。
            query: 検索クエリテキスト。
            top_k: 返す最大件数。
            where: ベクトルストアの where フィルタ。recall_with_identity から使用。
        """
        # リランク用に多めに候補を取得する
        fetch_k = top_k * 2
        results = self.vector_store.recall_inscribed_memory(query, character_id, fetch_k, where=where)

        now = datetime.now()

        # リランクとメタデータ注入
        reranked = []
        for mem in results:
            mem_id = mem.get("id")
            if not mem_id:
                continue

            try:
                m = self.sqlite.get_inscribed_memory(mem_id)
                if not m:
                    logger.debug(
                        "recall_memory: ベクトルストアにあるが SQLite に存在しない記憶をスキップ id=%s char=%s",
                        mem_id, character_id,
                    )
                    continue
                # soft-delete 済み記憶はスキップ（ベクトルストアから完全削除される前の過渡状態）
                if m.deleted_at is not None:
                    logger.debug(
                        "recall_memory: soft-delete 済み記憶をスキップ id=%s char=%s",
                        mem_id, character_id,
                    )
                    continue

                # 真の減衰スコアを計算する
                decayed_score = self.calculate_decayed_score(m, now)

                # cosine distance: 0=identical, 2=opposite。distance を similarity に変換する
                semantic_similarity = distance_to_similarity(mem.get("distance", 1.0))

                # ハイブリッドスコア: 意味的類似度 + 減衰スコア
                hybrid_score = (semantic_similarity * 0.5) + (decayed_score * 0.5)

                mem["hybrid_score"] = hybrid_score
                mem["decayed_score"] = decayed_score
                mem["semantic_similarity"] = semantic_similarity

                if m.created_at:
                    mem.setdefault("metadata", {})["created_at"] = m.created_at.isoformat(timespec="seconds")

                reranked.append(mem)
            except Exception:
                pass

        # ハイブリッドスコア降順でソートする
        reranked.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        final_results = reranked[:top_k]

        # 想起回数をインクリメント（last_accessed_at は更新しない — decay タイマー保持のため）
        for mem in final_results:
            mem_id = mem.get("id")
            if mem_id:
                self.sqlite.remember_inscribed_memory(mem_id)

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
            character_id: キャラクター ID。
            query: 検索クエリテキスト。
            identity_top_k: identity カテゴリから返す最大件数。
            other_top_k: identity 以外のカテゴリから返す最大件数。

        Returns:
            ``(identity_memories, other_memories)`` のタプル。
        """
        identity_memories = self.recall_inscribed_memory(
            character_id=character_id,
            query=query,
            top_k=identity_top_k,
            where={"category": "identity"},
        )
        logger.info(
            "recall_with_identity: identity=%d char=%s query=%.40s",
            len(identity_memories), character_id, query,
        )

        other_memories = self.recall_inscribed_memory(
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
        """記憶コレクションとチャット履歴を横断して検索する（PowerRecall 用）。

        通常の recall_memory と異なり、以下が異なる:
          - 時間減衰スコアによるリランクを行わない（意味的類似度のみで返す）
          - 論理削除レコードも除外しない（ベクトルストア側の管理のみ）
          - 記憶テーブルとチャット履歴テーブルを両方検索する

        Args:
            character_id: キャラクター ID。
            query: 検索クエリテキスト。
            top_k: 各コレクションから取得する最大件数。

        Returns:
            ``{"inscribed_memories": [...], "chat_turns": [...]}`` の辞書。
            各リストは id / content / distance / metadata を持つ dict。
        """
        memories = self.vector_store.recall_inscribed_memory(query, character_id, top_k=top_k)
        chat_turns = self.vector_store.recall_chat_turns(query, character_id, top_k=top_k)

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

        return {"inscribed_memories": memories, "chat_turns": chat_turns}

    @staticmethod
    def _build_context_window(msgs: list, message_id: str, window: int) -> list[dict]:
        """セッションのメッセージリストから指定 ID を中心にウィンドウを切り出す。

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

    def delete_inscribed_memory(self, memory_id: str, character_id: str) -> bool:
        """SQLite ソフトデリート + LanceStore ハードデリートを行う。

        SQLite を先に確定させ、LanceStore 削除はその後同期実行する。
        LanceStore 削除は原子的なので、過去のリトライキュー機構は不要。

        Args:
            memory_id: 削除対象の記憶ID。
            character_id: 所属キャラクターID（LanceStore の名前空間に使用）。

        Returns:
            SQLite の削除成否。False の場合は記憶が存在しないか既に削除済み。
        """
        ok = self.sqlite.soft_delete_inscribed_memory(memory_id)
        if ok:
            # SQLite コミット完了後に LanceStore から削除する
            self.vector_store.delete_inscribed_memory(memory_id=memory_id, character_id=character_id)
        return ok

    def restore_memory(self, memory_id: str) -> bool:
        """ソフトデリート済み記憶を復元する。

        LanceStore は復元対象外（再インデックスは write_inscribed_memory を使用）。
        SQLite の deleted_at を NULL に戻すだけ。

        Args:
            memory_id: 復元対象の記憶 ID。

        Returns:
            復元成否。False の場合は記憶が存在しないか削除されていない。
        """
        return self.sqlite.restore_inscribed_memory(memory_id)

    def delete_character_with_inscribed_memories(self, character_id: str) -> bool:
        """キャラクターと、紐づく全データをカスケード削除する。

        SQLite = source of truth のため、SQLite の削除を先に確定させてから
        LanceStore（ベクトル）の該当キャラ行を削除する。

        削除対象（SQLite は delete_character_cascade が担当）:
        - SQLite: characters / inscribed_memories / working_memory_threads(+posts) /
          session_drifts / 1on1 chat_sessions(+messages/images)
        - LanceDB: inscribed_memories / working_memory_threads / chat_turns / definitions

        保持: debug_log_entries（監査記録）。

        api/characters.py と api/ui.py の2箇所から共通利用するためにここで一元管理する。

        Args:
            character_id: 削除対象のキャラクターID。

        Returns:
            SQLite の削除成否。False の場合はキャラクターが存在しない。
        """
        result = self.sqlite.delete_character_cascade(character_id)
        if result:
            # SQLite の削除が確定してから LanceStore の各テーブルの該当キャラ行を削除する
            self.vector_store.delete_all_inscribed_memories(character_id)
            self.vector_store.delete_all_working_memory_threads(character_id)
            self.vector_store.delete_all_chat_turns(character_id)
            self.vector_store.delete_definition(character_id)
        return result

    def list_inscribed_memories(
        self,
        character_id: str,
        category: str | None = None,
        include_deleted: bool = False,
        sort_by: str = "created_at",
    ) -> list[dict]:
        """キャラクターの保存記憶一覧を dict リストで返す。"""
        mems = self.sqlite.list_inscribed_memories(character_id, category, include_deleted, sort_by)
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
