# 音声添付 仕様書 — 曲を聴かせる

> 状態: 未着手（設計合意済み 2026-08-20）
> 前提コミット: `b43889d`（claude_cli の画像対応。stream-json 入力へ一本化済み）

## 概要

キャラクターに **曲（音声ファイル）を聴かせられる** ようにする。歌詞抽出や採譜のような
前処理を挟まず、音声そのものを Gemini（`google` プロバイダー）へ渡す。

同時に、現状 backend 全体へ染み込んでいる **「添付＝画像」という前提を汎用化** する。
テーブル名・API パス・内部表現・フロントの型がすべて `image` を名乗っており、
ここに mp3 を流し込むと名前が嘘になる。

音声を扱えるのは Gemini だけで、Claude（`claude_cli` 経由含む）は音声入力を持たない
（実測済み。§棄却した設計案 4）。したがって **プロバイダーによって「聴ける／聴けない」が割れる**。
これはユーザの哲学（モデル切替＝キャラクターとの通信経路の変更であり、キャラクター本体は
Chotgor 側の記憶にあるのでズレない）のもとで受け入れる。聴かせたいときは Gemini へ切り替える。

## 要件（2026-08-20 合意）

1. **曲そのものを渡す**。文字起こし・採譜などの変換は行わない
2. **添付は最新ターンのみ送信**。過去ターンの添付は送らない（また聴かせたければ再送する）
3. **テーブル・API・内部表現の「画像」前提を汎用化する**（`chat_images` に mp3 を入れない）
4. **非対応プロバイダーでは入口で止める**。FileDialog の `accept` で絞り、
   それをすり抜けた場合も送信前に弾く（スマホの FileDialog は `accept` を尊重しないことがある）
5. 黙って捨てない。「渡らないのに渡ったように見える」状態を作らない

## 前提: 現状の「添付＝画像」構造

実装前に把握しておくべき現状。すべて画像専用になっている。

| 層 | 現状 |
|---|---|
| FileDialog | `MessageInput.tsx:215` `accept="image/*"` |
| プレビュー | `MessageInput.tsx:145` `URL.createObjectURL(file)` を `<img>` で描画 |
| アップロード呼び出し | `api/chat.ts:194` `uploadImages()` → `POST /api/chat/sessions/{id}/images` |
| アップロードAPI | `api/chat_images.py:34` `content_type` が `image/` 始まりでなければ 400 |
| 配信API | `api/chat_images.py:56` `GET /api/chat/images/{id}` |
| ORM | `models.py:94` `ChatImage`（table `chat_images`）／`models.py:70` `ChatMessage.images`(JSON) |
| ストア | `chat_store.py:365` `create_chat_image` / `get_chat_image` / `list_chat_images_by_session` |
| 送信 | `api/chat.py:435` `build_message_content(body.content, body.image_ids, ...)` |
| 内部表現 | `services/chat/content.py` → OpenAI vision 形式 `image_url` の data URL |
| 履歴組み立て | `content.py` `build_1on1_history` が**全履歴の user メッセージに画像を埋める** |
| 表示 | `ChatBubbles/images.tsx` が `/api/chat/images/{id}` を `<img>` で描画 |
| セッション削除 | `api/chat.py:211` uploads_dir 内のファイルを消す |

### 重要 — 添付の寿命は共通仕様になっていない

「最新ターンのみ」は **`claude_cli` プロバイダー内部の実装**（`_extract_latest_images`、
2026-08-19 実装）であって、Chotgor 共通の仕様ではない。
`google_provider._build_contents`（`google_provider.py:224`）は
**履歴に含まれる全画像を毎ターン送っている**。

このまま音声を入れると、Gemini へ曲が毎ターン再送される（4分の曲で毎回7000トークン強）。
**音声対応の前に、寿命ルールを `content.py`（プロバイダー共通の層）へ引き上げる**こと。
Phase 2 がこれに当たる。

## 設計

### 全体フロー

```
ユーザが mp3 を添付
  ↓ フロント: 選択中プリセットの attachment_kinds に audio が無ければ選ばせない／弾く
POST /api/chat/sessions/{id}/attachments   （MIME ホワイトリスト検証）
  ↓ uploads_dir/{id} に保存、chat_attachments に mime_type を記録
POST /api/chat/sessions/{id}/stream  { attachment_ids: [...] }
  ↓ build_message_content: 最新ターンの添付を content パートへ
  ↓ build_1on1_history:   過去ターンの添付はテキスト痕跡へ置換
google_provider._build_contents  → types.Part.from_bytes(mime_type="audio/mpeg")
  ↓
Gemini が曲を聴く → 応答 → キャラクター自身が inscribe_memory で印象を残す
```

### ① データ

- テーブル `chat_images` → **`chat_attachments`**（`ALTER TABLE ... RENAME TO`）
- ORM クラス `ChatImage` → **`ChatAttachment`**（`models.py`）
- 列 `chat_messages.images`(JSON) → **`attachments`**（`ALTER TABLE ... RENAME COLUMN`、SQLite 3.25+）
- **`mime_type` 列は既存のものをそのまま使う**。種別（image / audio）は mime から導出する。
  `kind` 列は追加しない（導出できるものを持たない）
- ファイル実体は従来どおり `uploads_dir/{id}`（拡張子なし）

マイグレーションは冪等メソッドを1つ追加し、`store.py` の `__init__` 末尾（`_migrate_add_bubble_color()` の後）へ登録する:

```python
def _migrate_rename_chat_images_to_attachments(self) -> None:
    """chat_images → chat_attachments、chat_messages.images → attachments へリネーム。

    添付が画像だけではなくなったため（音声）。既に新名なら何もしない。冪等。
    """
```

`sqlite_master` を見て旧名が存在するときだけ `ALTER TABLE` を撃つこと
（`_migrate_rename_initiative_cap` `migrations.py:1416` が書き方の参考になる）。

> **マイグレーション前に `data/chotgor.db` と `data/lancedb/` を退避すること。**

### ② API

| 旧 | 新 |
|---|---|
| `POST /api/chat/sessions/{id}/images` | `POST /api/chat/sessions/{id}/attachments` |
| `GET /api/chat/images/{id}` | `GET /api/chat/attachments/{id}` |
| リクエスト `image_ids` | `attachment_ids`（`api/schemas.py`・`api/chat.py`・`api/chat.ts`） |
| レスポンス `images` | `attachments`（`api/utils.py:96` `message_to_dict`） |
| ファイル `api/chat_images.py` | `api/chat_attachments.py`（`main.py` の include も更新） |

受け入れ MIME はホワイトリストを **1箇所の定数**に置く（`api/chat_attachments.py`）:

```python
# Gemini が inline_data で受け取れる音声形式に合わせる。
_ALLOWED_AUDIO = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/flac", "audio/aac"}
# 画像は従来どおり image/* を全許可（Anthropic 側で弾かれる形式は provider が捨てる）
```

APIレイヤは「Chotgor が扱える添付種別か」だけを見る。プロバイダー適合（音声を渡せるか）は
セッションのプリセット次第で変わるため、送信時に判定する（§④）。

### ③ 内部メッセージ表現

音声パートは **OpenAI 準拠の `input_audio` 形式**を採用する:

```python
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "mp3"}}
```

- 画像は現状どおり `image_url` の data URL（変更しない）
- `format` は mime から導出（`audio/mpeg` → `"mp3"`）
- 採用理由: 独自形式を作らずに済み、将来 `openai_provider` が音声対応したときそのまま乗る。
  Anthropic 風の `{"type":"audio","source":{...}}` に寄せる理由はない（Anthropic は音声非対応）

`content.py` の `build_message_content` を、mime_type を見てパートを出し分ける形へ拡張する。

### ④ 添付の寿命 — 最新ターンのみ

`content.py` の `build_1on1_history` で、**最新ターン以外の添付をテキスト痕跡へ置換**する。

```
（過去ターン）「これ聴いて」＋mp3  →  「これ聴いて\n[音声を聴かせた]」
（最新ターン）「これ見て」＋png    →  content パートとして実体を載せる
```

- これはプロバイダー共通の層なので、google / claude_cli / その他すべてに効く
- `claude_cli` 側の `_extract_latest_images` はそのまま残す（二重の安全網）
- 痕跡テキストにファイル名を出したいが、**現状ファイル名は DB に保存していない**
  （`chat_attachments` に `filename` 列がない）。名前を出すなら列追加が必要 → §将来枠

キャラクターにとって、曲は「聴かせた瞬間」にだけ存在する。印象を残すかどうかは
キャラクター自身が `inscribe_memory` で決める（忘却は機能である）。

### ⑤ プロバイダー能力と入口ガード

`BaseLLMProvider` にクラス属性で能力を宣言する:

```python
# このプロバイダーが受け取れる添付種別。UI の入口ガードと送信時検証の唯一の根拠。
SUPPORTED_ATTACHMENT_KINDS: set[str] = {"image"}
```

| プロバイダー | 値 | 根拠 |
|---|---|---|
| `google` | `{"image", "audio"}` | Gemini は inline_data で音声を受ける |
| `claude_cli` | `{"image"}` | 実測で audio/mpeg は API が拒否 |
| `anthropic` / `openai` / `openrouter` / `xai` / `ollama` / `sakura` | 実態に合わせる（不明なら `{"image"}`） | — |

- `/v1/models`（`adapters/openai/router.py:132`）のレスポンス dict に
  `"attachment_kinds": [...]` を追加する。`registry` から provider クラスを引いて属性を読む
- フロント `Model` 型（`api/chat.ts:29`）に `attachment_kinds?: string[]` を追加
- `App.tsx` は `selectedModel` に対応する `Model` を持っているので、そこから
  `ChatView` → `MessageInput` へ渡す
- `MessageInput` は `accept` を動的生成し、**選択後にも MIME を検査**して非対応なら弾く
- **backend 側の二重ガード**: 送信API（`api/chat.py` の stream）で、選択プロバイダーが
  非対応の添付種別を含むリクエストを受けたら 400 で拒否する。黙って捨てない

### ⑥ フロント表示

- `MessageInput` のプレビュー: 画像はサムネのまま、音声は**ファイル名チップ＋`<audio controls>`**
  （`URL.createObjectURL` を `<img>` に食わせると音声で壊れる）
- `ChatBubbles/images.tsx` → `attachments.tsx` へ汎用化。音声は
  `<audio controls src="/api/chat/attachments/{id}">`

## フェーズ分け

各フェーズの完了条件は **`python -m pytest` 全通過**（着手時点で 2198 件）。

| Phase | 内容 | 挙動変化 |
|---|---|---|
| 1 | 汎用化リネーム（DB・API・内部・フロントの命名を attachment へ） | なし（純粋なリネーム） |
| 2 | 添付寿命を `content.py` へ引き上げ（最新ターン以外は痕跡テキスト化） | **あり** — google で履歴画像が送られなくなる |
| 3 | 音声の受け入れ（MIME 拡張・`input_audio`・`google_provider` の Part 化） | 音声が届くようになる |
| 4 | 入口ガード（`attachment_kinds`・`accept` 動的化・送信時 400） | 非対応プロバイダーで添付が止まる |
| 5 | 表示（`<audio>` プレイヤー・プレビュー） | 音声が再生できる |

Phase 1 と 2 は独立して価値がある（2 は現状の Gemini のトークン浪費を止める）。
Phase 3 以降が音声本体。

## 改修ファイル見積もり

| ファイル | Phase | 変更 |
|---|---|---|
| `backend/repositories/sqlite/models.py` | 1 | `ChatImage`→`ChatAttachment`、`images`→`attachments` |
| `backend/repositories/sqlite/migrations.py` | 1 | リネーム用の冪等マイグレーション追加 |
| `backend/repositories/sqlite/store.py` | 1 | `__init__` へマイグレーション登録 |
| `backend/repositories/sqlite/stores/chat_store.py` | 1 | メソッド名・参照の追随 |
| `backend/api/chat_images.py` → `chat_attachments.py` | 1,3 | リネーム＋MIME ホワイトリスト |
| `backend/main.py` | 1 | ルーター include の更新 |
| `backend/api/chat.py` | 1,4 | `image_ids`→`attachment_ids`、送信時ガード |
| `backend/api/schemas.py` / `utils.py` | 1 | フィールド名 |
| `backend/services/chat/content.py` | 1,2,3 | パート出し分け・寿命ルール |
| `backend/providers/base.py` | 4 | `SUPPORTED_ATTACHMENT_KINDS` |
| `backend/providers/google_provider.py` | 3 | `_build_contents` に音声 Part |
| `backend/providers/claude_cli_provider.py` | 4 | 能力宣言のみ（音声は既に無視される） |
| `backend/adapters/openai/router.py` | 4 | `/v1/models` に `attachment_kinds` |
| `frontend/src/api/chat.ts` | 1,4 | `uploadImages`→`uploadAttachments`、`Model` 型 |
| `frontend/src/App.tsx` | 4 | 選択中プリセットの能力を下へ渡す |
| `frontend/src/components/ChatView.tsx` | 4 | props 中継 |
| `frontend/src/components/MessageInput.tsx` | 4,5 | `accept` 動的化・MIME 検査・音声プレビュー |
| `frontend/src/components/ChatBubbles/images.tsx` → `attachments.tsx` | 1,5 | 汎用化・`<audio>` |
| `frontend/src/components/ScenarioChatView/index.tsx` | 4 | `MessageInput` の props 追随 |
| `docs/current-spec/ARCHITECTURE.md` | 全 | 添付の扱いを追記 |

## 棄却した設計案（検討記録）

1. **音声を文字起こししてテキストで渡す**（Whisper／Demucs+Whisper で歌詞抽出／
   librosa 等で BPM・キー抽出／audio captioning で説明文）
   → プロバイダー非依存になる利点はあるが、**曲を聴かせる用途では情報が落ちる**。
   歌詞だけならキャラクターが受け取るのは詩のテキストで、インストでは渡すものが無くなる。
   ユーザの哲学（モデル切替＝通信経路の変更）に照らせば、Gemini へ切り替えて
   直接聴かせる方が素直だと判断した。
2. **MIDI／楽譜形式（ABC記法・MusicXML）へ変換して渡す**
   → 混合音源の自動採譜は実用に遠く（単音ピアノならある程度）、
   MIDI のノート列や ABC 記法を LLM へ渡しても音楽としては読み取りにくい。
3. **音声を数ターン保持する**
   → Gemini の音声はおおむね 1 秒 32 トークン。4 分の曲で 7000 トークン強を毎ターン再送する。
   最新ターンのみとし、印象はキャラクター自身が `inscribe_memory` で残す形にした。
4. **`claude_cli` でも音声を渡す**
   → 実測で不可能（2026-08-19 検証）。`document` ブロックに `audio/mpeg` を載せると
   API が「PDF ではない形式または破損したファイル」として拒否する。`document` は PDF 専用。
5. **`chat_images` テーブルのまま音声を入れる**
   → マイグレーションは省けるが、テーブル名が嘘になる。Chotgor の命名規則（役割を名前に出す）に反する。
6. **`kind` 列を追加して種別を持つ**
   → `mime_type` から導出できる。持たなくて済むものを持たない。
7. **非対応プロバイダーでは添付を黙って捨てる**
   → 「渡ったように見えて渡っていない」が最悪。画像で同じことをやって（claude_cli の
   「見えません」注記）2026-08-19 に解消したばかり。入口で止める。

## テスト観点

- マイグレーションの冪等性（2回実行しても壊れない／既存行が保持される／旧名が無い DB でも安全）
- `build_1on1_history`: 最新ターン以外の添付が痕跡テキストへ置換される（画像・音声とも）
- `build_message_content`: mime に応じて `image_url` / `input_audio` を出し分ける
- `google_provider._build_contents`: `input_audio` が `types.Part` へ変換される
- `claude_cli`: 音声パートが混ざっても発話が成立する（黙って捨てる／例外を出さない）
- アップロードAPI: ホワイトリスト外の MIME は 400
- 送信API: 非対応プロバイダー＋音声添付は 400
- `/v1/models` が `attachment_kinds` を返す

## 将来枠

- **PDF**: `claude_cli` 経由で Anthropic の `document` ブロックが通ることは実測済み
  （2026-08-19、PDF 内テキストの読み取りに成功）。同じ添付抽象へ乗せられる
- **ファイル名の保持**: `chat_attachments.filename` 列を足すと、痕跡テキストが
  `[音声を聴かせた: kalafina_hikari.mp3]` のように具体的になる
- **動画**: Gemini は対応。同じ経路に乗る
- **サイズ・コスト**（実装時に要確認）: Gemini の inline data はリクエスト 20MB 前後が上限で、
  超過分は File API 経由になる。Chotgor は base64 を毎回埋め込むため、
  大きいファイルはメモリとレイテンシに効く
