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
import threading
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

# 一覧注入（システムプロンプト）に載せる Close 済みスレッドの既定上限。
# Close は決着済みだが件数が増え続けるためプロンプトを単調に圧迫する。直近ぶんだけを
# 一覧へ残し、それ以前は read_working_memory_list ツールで本人が取りに行く形にする
# （current-spec/memory_recall_algorithm.md §4.3）。
DEFAULT_CLOSED_INDEX_LIMIT = 30

# heat 想起で切り捨てる下限。関連の薄いスレッドまで前景へ上がるのを防ぐ。
# 該当なし（0件）のターンがあってよい。
# heat の relevance 項は cosine 類似度そのもので、値域が embedding モデルに強く依存する
# （現行 bge-m3 は無関係で約0.41・関連しても0.55前後にしか伸びない）。旧 0.05 は実質
# 「importance × decay >= 0.1」を要求する閾値として働き、topic は数日で前景へ上がらなく
# なっていた。**モデルを変えたらこの値も見直すこと**（current-spec/memory_recall_algorithm.md §2.1）。
DEFAULT_WM_RECALL_MIN_HEAT = 0.03

# heat 想起でベクトル検索から取り寄せる候補数（リランク前）。
# ベクトル検索の順序は relevance 順であって heat 順ではないため、狭く取ると
# 「importance × decay が高いのに相槌ターンで relevance が伸びないスレッド」が
# 評価される前に落ちる。実測では heat 上位3件が relevance 順で12〜16位に沈んでいた。
# Open スレッドは数十本規模なので、ほぼ全件を評価しても検索コストは無視できる。
WM_RECALL_FETCH_MULTIPLIER = 10
WM_RECALL_FETCH_MIN = 30

# Open × Close の重複疑い検出（find_similar_closed_threads）で使う relevance 下限。
# はるの実データ実測（docs/planned/wm_repeat_awareness_plan.md）では
# 無関係ペア 0.776 / 明確な重複 0.834〜0.845 / 同一トピック内 0.874 だった。
# その間に置いた初期値であり、1キャラ・9ペアの実測にすぎない。誤検出が頻発するなら
# 上げ、拾ってほしいものを見逃すなら下げる前提の運用値。
DEFAULT_SIMILAR_CLOSED_MIN_RELEVANCE = 0.82


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
    # close できる種別（解決を目指す task/topic のみ）。emotion/body/relation は
    # 「自然に消える」対象であり明示的な close 操作はできない — 更新のみ許可する。
    # relation を close すると圧力エンジンの関係重み（compute_social）が既定値へ
    # 落ちる副作用があり、キャラクター本人にその副作用は見えないため事故を防ぐ。
    _CLOSABLE_TYPES = frozenset({"task", "topic"})
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

    def _reindex_thread_sync(self, thread_id: str) -> None:
        """スレッドの LanceStore embedding を最新状態で upsert する（同期実体）。

        embedding 生成（infinity への HTTP 呼び出し）を含むため遅い。
        ツール応答のクリティカルパスから外すため、通常は ``_reindex_thread()`` 経由で
        バックグラウンドスレッドから呼ぶこと。直接呼ぶのはテスト等の同期完了を要する場合のみ。
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

    def _reindex_thread(self, thread_id: str) -> None:
        """スレッドの LanceStore embedding 更新をバックグラウンドで実行する（fire-and-forget）。

        SQLite が source of truth なので、index 更新は遅延・失敗してもキャラの応答品質に
        即時影響しない（次回更新時に最新化される）。embedding サーバ（infinity）の
        詰まりが MCP の 30 秒タイムアウトに乗らないよう、必ず別スレッドへ逃がす。

        例外はスレッド内で握り潰して warning ログに残す。daemon=True なのでプロセス終了時に
        強制終了され、index は次の更新で最新化される（ベストエフォート）。
        """
        def _run() -> None:
            try:
                self._reindex_thread_sync(thread_id)
            except Exception as e:
                logger.warning(
                    "WM reindex 失敗 thread_id=%s err=%s: %s",
                    thread_id, type(e).__name__, e,
                )

        threading.Thread(
            target=_run,
            name=f"wm-reindex-{thread_id[:8]}",
            daemon=True,
        ).start()

    def _thread_to_dict(self, thread, include_posts: bool = False,
                         include_latest_post: bool = False) -> dict:
        """スレッド ORM を注入・API 用の dict に変換する。

        Args:
            thread: WorkingMemoryThread ORM オブジェクト。
            include_posts: True なら全ポストを ``posts`` キーに含める（read_working_memory_thread 用）。
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
            # 由来タグ。recall 表示で "real" / "usual" / "interlude" を区別するために載せる。
            "origin": getattr(thread, "origin", "real") or "real",
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
        relation_target: str | None = None,
        content: str | None = None,
        origin: str = "real",
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
            origin: スレッドのソース識別（3値）。"real"=日常、"usual"=うつつ（ユーザ未共有の自分の生活体験）、"interlude"=シナリオPCモードの幕間。

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
            origin=origin,
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
        summary: str | None = None,
        atmosphere_tag: str | None = None,
        importance: float | None = None,
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
        """スレッドの is_open フラグを更新する。

        close（is_open=False）は _CLOSABLE_TYPES（task/topic）のスレッドのみ許可する。
        emotion/body/relation は「自然に消える」ものであり明示的に閉じられない
        （更新のみ可能）。reopen（is_open=True）は種別を問わず常に許可する。

        Returns:
            更新成否。

        Raises:
            ValueError: close 対象のスレッドが _CLOSABLE_TYPES 以外の種別の場合。
        """
        if not is_open:
            thread = self.sqlite.get_working_memory_thread(thread_id)
            if thread and thread.type not in self._CLOSABLE_TYPES:
                raise ValueError(
                    f"'{thread.type}' 型スレッドは close できません（更新のみ可能）。"
                )
        ok = self.sqlite.update_working_memory_thread(thread_id, is_open=is_open)
        if ok:
            self._reindex_thread(thread_id)
        return ok

    # ------------------------------------------------------------------
    # 参照（システムプロンプト注入 / ツール）
    # ------------------------------------------------------------------

    def resolve_thread_id(self, character_id: str, id_or_prefix: str) -> str | None:
        """短縮 ID（前方一致）をフルスレッド ID へ解決する。

        システムプロンプトのスレッド一覧はトークン節約のため短縮 ID（先頭8桁）で
        表示される。キャラクターがツールへ短縮 ID を渡してきたとき、この関数で
        フル ID に復元する。フル ID がそのまま渡された場合も従来どおり通す。

        Args:
            character_id: 前方一致検索のスコープとなるキャラクター ID。
            id_or_prefix: フル ID または短縮 ID。

        Returns:
            解決済みフル ID。該当なしなら None。

        Raises:
            ValueError: 前方一致が複数スレッドに衝突した場合。
        """
        if not id_or_prefix:
            return None
        # フル ID の完全一致を優先（Chronicle・UI 等の内部経路は常にこちら）
        if self.sqlite.get_working_memory_thread(id_or_prefix) is not None:
            return id_or_prefix
        matches = self.sqlite.find_working_memory_thread_ids_by_prefix(
            character_id, id_or_prefix
        )
        if len(matches) > 1:
            raise ValueError(
                f"ID '{id_or_prefix}' は複数のスレッドに一致します。より長い ID を指定してください"
            )
        return matches[0] if matches else None

    def list_all_threads(
        self,
        character_id: str,
        closed_limit: int | None = DEFAULT_CLOSED_INDEX_LIMIT,
    ) -> tuple[list[dict], int]:
        """一覧注入用のスレッドと、一覧から省いた Close 済み本数を返す。

        self_history 代替の「全スレッド一覧」注入に使う。最新ポストは含めない。
        Open は常に全件、Close 済みは updated_at 降順で closed_limit 本までに絞る。
        省いた本数は告知行に使い、本体は read_working_memory_list で取りに行ける
        （視界から消すのではなく「存在は見えていて中身は開いて読む」形にする）。

        Args:
            character_id: キャラクター ID。
            closed_limit: 一覧に載せる Close 済みの上限本数。None なら全件（省略数は 0）。

        Returns:
            (スレッド dict リスト（updated_at 降順）, 一覧から省いた Close 本数)。
        """
        threads = self.sqlite.list_working_memory_threads(character_id)
        if closed_limit is None:
            return [self._thread_to_dict(t) for t in threads], 0
        kept = []
        closed_seen = 0
        omitted = 0
        for t in threads:
            if t.is_open:
                kept.append(t)
                continue
            closed_seen += 1
            if closed_seen <= closed_limit:
                kept.append(t)
            else:
                omitted += 1
        return [self._thread_to_dict(t) for t in kept], omitted

    def get_fixed_threads(
        self,
        character_id: str,
        participants: list[str] | None = None,
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

    def get_thread_detail(self, thread_id: str) -> dict | None:
        """スレッド1件＋全ポストを dict で返す（read_working_memory_thread ツール用）。

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
        type: str | None = None,
        is_open: bool | None = None,
        include_latest_post: bool = True,
    ) -> list[dict]:
        """type / is_open で絞り込んだスレッド一覧を返す（Chronicle・UI・一覧ツール用）。

        Args:
            include_latest_post: False なら最新ポスト本文を含めない。見出しだけを並べる
                read_working_memory_list 用（ポスト本文はそのまま出すと長大なため）。
        """
        threads = self.sqlite.list_working_memory_threads(character_id, type=type, is_open=is_open)
        return [
            self._thread_to_dict(t, include_latest_post=include_latest_post)
            for t in threads
        ]

    def recall_threads(
        self,
        character_id: str,
        query: str,
        top_k: int = 3,
        min_heat: float = DEFAULT_WM_RECALL_MIN_HEAT,
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
            min_heat: この heat 未満のスレッドは前景に上げない（0件のターンがあってよい）。

        Returns:
            heat 降順のスレッド dict リスト（``heat`` キー付き、最新ポスト込み）。
        """
        # heat 計算でリランクするため多めに取得する（狭く取ると heat 上位が
        # relevance 順の下位に沈んで脱落する。定数の説明を参照）
        fetch_k = max(top_k * WM_RECALL_FETCH_MULTIPLIER, WM_RECALL_FETCH_MIN)
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
            if heat < min_heat:
                continue
            d = self._thread_to_dict(thread, include_latest_post=True)
            d["heat"] = heat
            scored.append(d)
        scored.sort(key=lambda x: x.get("heat", 0.0), reverse=True)
        return scored[:top_k]

    def find_similar_closed_threads(
        self,
        character_id: str,
        open_thread: dict,
        top_k: int = 1,
        min_relevance: float = DEFAULT_SIMILAR_CLOSED_MIN_RELEVANCE,
    ) -> list[dict]:
        """Open スレッド1件に対し、意味的に近い Close 済みスレッドを検索する。

        Chronicle 棚卸し時に「実はもう Close 済みの話題と同じ内容を Open のまま
        抱え続けていないか」をキャラクター本人が気づくための参考情報を作る。
        ここでは close を実行しない（判断は本人に委ねる。close するかどうかの
        決定は thread_updates.is_open を通じて本人の応答に委ねる）。

        recall_threads と骨格は同じだが、向きが逆になっている点に注意:
        recall_threads は「クエリ文字列 → Open スレッド」、こちらは
        「Open スレッド → Close スレッド」なので、クエリ自体を Open 側スレッドの
        index テキスト（summary + 最新ポスト）から組み立てる。

        Args:
            character_id: キャラクター ID。
            open_thread: 対象の Open スレッド dict（summary / latest_post を含む。
                list_threads_by_type の戻り値をそのまま渡せる）。
            top_k: 返す最大件数。
            min_relevance: この relevance 未満は「重複の疑いなし」として除外する
                （distance_to_similarity 後のスケール。既定値の根拠は
                docs/planned/wm_repeat_awareness_plan.md 参照）。

        Returns:
            relevance 降順のスレッド dict リスト（``relevance`` キー付き）。
        """
        query_text = (open_thread.get("summary") or "").strip()
        latest = open_thread.get("latest_post") or ""
        if latest:
            query_text = (query_text + "\n" + latest).strip()
        if not query_text:
            return []

        fetch_k = max(top_k * 2, top_k)
        results = self.vector_store.recall_working_memory_threads(
            query_text,
            character_id,
            top_k=fetch_k,
            where={"type": {"$in": ["task", "topic"]}, "is_open": 0},
        )
        scored: list[dict] = []
        for r in results:
            thread = self.sqlite.get_working_memory_thread(r.get("id", ""))
            if not thread or thread.is_open:
                continue
            # distance 欠落時は 2.0（対極）に倒し、relevance 0 として弾く。
            relevance = distance_to_similarity(r.get("distance", 2.0))
            if relevance < min_relevance:
                continue
            d = self._thread_to_dict(thread, include_latest_post=True)
            d["relevance"] = relevance
            scored.append(d)
        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored[:top_k]
