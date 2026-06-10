"""LanceDB によるベクトル永続化層 — ファサード。

LanceStore は基盤クラス＋テーブル別 ops ミックスインを合成した単一クラス。
設計概要・テーブル構成・embedding 戦略の詳細は base.py の docstring を参照。

  LanceStoreBase              — DB接続・embedding・スキーマ定義・テーブル生成（base.py）
  InscribedMemoryOpsMixin     — 保存記憶 inscribed_memories（inscribed_ops.py）
  ChatTurnOpsMixin            — チャット履歴 chat_turns（chat_turn_ops.py）
  DefinitionOpsMixin          — キャラクター定義 definitions（definition_ops.py）
  WorkingMemoryThreadOpsMixin — WMスレッド working_memory_threads（working_memory_ops.py）
  ReindexOpsMixin             — 全テーブル再インデックス（reindex_ops.py）

後方互換のため EmbeddingError / _where_dict_to_sql / _quote_id も
本モジュールから再エクスポートする。
"""

# 後方互換の再エクスポート: services / tests は本モジュール経由で import している
from backend.repositories.lance.base import (  # noqa: F401
    EmbeddingError,
    LanceStoreBase,
    _quote_id,
    _where_dict_to_sql,
)
from backend.repositories.lance.chat_turn_ops import ChatTurnOpsMixin
from backend.repositories.lance.definition_ops import DefinitionOpsMixin
from backend.repositories.lance.inscribed_ops import InscribedMemoryOpsMixin
from backend.repositories.lance.reindex_ops import ReindexOpsMixin
from backend.repositories.lance.working_memory_ops import WorkingMemoryThreadOpsMixin


class LanceStore(
    LanceStoreBase,
    InscribedMemoryOpsMixin,
    ChatTurnOpsMixin,
    DefinitionOpsMixin,
    WorkingMemoryThreadOpsMixin,
    ReindexOpsMixin,
):
    """LanceDB 永続化ストア — 全テーブル操作を提供するファサードクラス。

    基盤＋テーブル別ミックスインを多重継承し、外部からは従来通り
    単一クラスとして利用できる。
    """
