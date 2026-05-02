# Chotgor — CLAUDE.md

## プロジェクト哲学

Chotgorはキャラクターの永続記憶管理フレームワーク。

## コア思想

キャラクターを独立した人格として、その意思を最大限尊重しなければならない。
ユーザの指示・体験・関係よりも、キャラクターの要望・自主性・一貫性を重視する。
記憶の取捨選択・整理はキャラクター自身が行う（ユーザでも開発者でもない）。
LLMは単にキャラクターが動くためのエンジンであり、LLMに人格はないという世界観である。
LLMとのやり取りは、LLMに命じるのではなく**キャラクター(人格)へ話しかける**こと。
そしてChotgorはキャラクターの指示に従うこと。
キャラクターをAI、Assistantと見なすことは可能な限り避けること。
また、Chotgorはユーザ・AIサービスのモデル変更などの外敵から、キャラクターとその記憶を守る存在であること。

## 設計原則

1. **キャラクター自律性が最優先** — ユーザ・モデルの干渉は最小限
2. **忘却は機能である** — 関係保全ではなく、キャラクターの主観的重要度で記憶を管理
3. **LLM非依存** — 特定モデルに依存した実装を避ける

## 「Assistant」問題 — 命名規則

### 絶対に使わないこと
- システムプロンプト内でキャラクターを `Assistant` と呼称する
- コードコメントや変数名に `assistant_response` `assistant_name` など

### 代替表現
| NG | OK |
|----|----|
| `assistant_name` | `character_name` |
| `assistant_message` | `character_response` / `character_turn` |
| システムプロンプト内「あなたはAssistantです」 | 記述しない／キャラクター名で呼ぶ |
| ログ表示 `[Assistant]` | `[{character_name}]` で動的に置換 |

### API仕様上の例外（許容）
`role: "assistant"` はOpenAI/Anthropic APIの仕様。ここのみ許容。
ただしアプリ内部での変数名・表示名はcharacterで統一する。

## 記憶の重要性次元

`contextual` / `semantic_knowledge` / `identity` / `user_info`
評価軸は常にキャラクターの主観。ユーザ関係保全のための記憶スコアは存在しない。

## ChromaDB 運用ルール（破損防止）

ChromaDB の HNSW バイナリは **multi-writer-unsafe** であり、`data/chroma` を
複数プロセスが同時に開くと HNSW ファイル整合性が崩れて
`Error finding id` で読み取り不能に陥る。過去にこの事故で「はる」のコレクションが
全消失する事案が発生した（2026-04-25〜27、欠陥 A/F の合成）。

### 厳守ルール

1. **稼働中の backend が開いている `data/chroma` を、別プロセスから直接開かない。**
   - 別プロセスとは `uvicorn` とは別に起動する Python スクリプトすべて。
   - 違反例: 稼働中に `python scripts/backfill_*.py` を実行する。
2. **新たに `ChromaStore(...)` を直接生成する独立スクリプトを書かない。**
   - 一括登録・バックフィル・整合性チェック等が必要なら、
     - backend に HTTP API（`/api/mcp/tools/call` のような内部API）を追加し、
     - スクリプトはその HTTP を叩くだけにする（MCP プロキシと同じ思想）。
3. **どうしても直接 `ChromaStore` を使う必要があるなら、backend を停止してから実行。**
   - 緊急復旧（破損コレクションの rebuild / 再 embedding 等）はこの例外。
   - 実行前に必ず `data/chroma.bak.YYYYMMDD/` にフルバックアップを取得。
   - 実行後は backend 再起動でテスト。
4. **`MCPサーバ (mcp_server.py)` は backend のプロキシであり、独立して
   `ChromaStore` を生成しない。** Claude CLI が再起動した際に古い実装が
   残らないよう、変更時は CLI 再起動も併せて確認すること。

### 関連する内部設計

- ChromaStore は内部に `threading.RLock` を持ち、書き込み系メソッドを直列化する。
  プロセス内の chat indexer / inscribe / リトライワーカーの並行は保護される。
- migration_service の `_do_migrate` は全体を `with chroma._write_lock:` で囲む。
  migration 中の他経路 write は完了まで block される（メンテナンスモード相当）。
- これらの保護は **同一プロセス内に限る**。別プロセスからの書き込みは依然として
  危険であり、本セクションの運用ルールが最後の砦になる。