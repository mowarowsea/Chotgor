"""LanceDB ベクトルストアパッケージ。

Lance フォーマット（追記＋アトミックコミット型）のベクトル永続化層を提供する。
書き込み中のクラッシュでもインデックスが壊れない構造的堅牢性が特徴。
"""

from backend.repositories.lance.store import LanceStore  # noqa: F401
