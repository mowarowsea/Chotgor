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

## LanceDB 運用ルール

ベクトルストアは LanceDB（`data/lancedb`）に統一されている。
旧 ChromaDB の HNSW バイナリ破損問題（過去に「はる」のコレクション全消失事案あり）への
根治対応として、Lance フォーマット（追記＋アトミックコミット）に置き換えた。
旧 ChromaDB データの最終バックアップは `data/chroma.bak.YYYYMMDD/` に保管されている。

### 設計概要

- **テーブル構成（単一テーブル方式）**:
  - `memories`    — 記憶（旧 `char_{character_id}` 群を統合）
  - `chat_turns`  — チャット履歴（旧 `chat_{character_id}` 群を統合）
  - `definitions` — キャラクター定義（旧 `char_definitions` 相当）
- 単一テーブル + `character_id` カラムでフィルタする方式。
  キャラクター数の増加でテーブル数が爆発しない。
- 書き込みは `merge_insert` で原子的に upsert される。
  旧 ChromaStore で必要だった `_safe_get_or_create_collection` / `rebuild_memory_collection`
  / リトライキュー機構などはすべて撤廃済み。

### 運用ルール

1. **新たに `LanceStore(...)` を直接生成する独立スクリプトを書く前に、
   本当に独立スクリプトが必要か再検討すること。**
   - 一括登録・バックフィル・整合性チェック等は
     backend に HTTP API（`/api/mcp/tools/call` のような内部API）を追加して
     そちらを叩くのが望ましい（MCP プロキシと同じ思想）。
   - LanceDB は ChromaDB と違って multi-process write でも破損しないが、
     アプリ全体の単一インスタンス集約という設計利点は維持する。
2. **`MCPサーバ (mcp_server.py)` は backend のプロキシであり、独立して
   `LanceStore` を生成しない。** Claude CLI が再起動した際に古い実装が
   残らないよう、変更時は CLI 再起動も併せて確認すること。
3. **embedding model 変更時は Settings UI からの再インデックスを使う。**
   `LanceStore.reindex_all(new_embedding_fn, sqlite)` が SQLite を source of truth として
   全テーブルを drop → 再 embed → insert する。直接スクリプトで触らない。

### Embedding 設定

- `infinity` プロバイダー（`http://localhost:7997`、ruri-v3-310m など）を推奨。
- `google` プロバイダー（Gemini Embedding）も対応。
- ChromaDB 組み込みの `default` プロバイダーは LanceStore で非サポート
  （sentence-transformers 依存を持ち込まない方針）。