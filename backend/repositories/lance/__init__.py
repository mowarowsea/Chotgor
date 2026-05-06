"""LanceDB ベクトルストアパッケージ。

ChromaStore の置き換えとして、HNSW バイナリ破損問題を構造的に回避する LanceDB ベースの
ベクトル永続化層を提供する。Lance フォーマットは追記＋アトミックコミット型のため、
書き込み中のクラッシュでもインデックスが壊れない。
"""

from backend.repositories.lance.store import LanceStore  # noqa: F401
