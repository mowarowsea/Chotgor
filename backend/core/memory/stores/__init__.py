"""SQLiteStore ドメイン別 Mixin パッケージ。

各 Mixin クラスは self.get_session() を SQLiteStore から継承して使用する。
SQLiteStore は全 Mixin を多重継承することで後方互換を維持する。
"""
