# 1on1 の引き直し結果を ◀ ▶ で選び直す（1on1 レスポンスガチャ）

> ステータス: **実装完了（2026-09-28）**。

## 目的

1on1 チャットで引き直し（再生成）した応答を捨てずに残し、シナリオと同じ
`◀ 2/3 ▶` の枝ナビで選び直せるようにする。

## 要件（絞り込み済み）

1. **切り替えられるのは最新ターンの応答だけ**。
2. 選ばれなかった応答は**別テーブル `chat_message_variants` へ退避**する。
   `chat_messages` は従来どおり「本線の一直線」のまま。

## 採用した設計 — 別テーブル退避方式

### テーブル `chat_message_variants`

`chat_messages` と同じ列を持ち、次の3列を足す。

| 列 | 意味 |
|----|------|
| `anchor_message_id` | その応答が返事をしていたユーザ発話の ID（兄弟判定のキー） |
| `variant_group_id` | 1回の応答（＝起点ユーザ発話より後の行の束）をまとめるキー |
| `archived_at` | 退避した時刻 |

主キー `id` は元の `chat_messages.id` をそのまま使う。戻すときに同じ ID で
`chat_messages` へ戻し、退避側の行は消すので衝突しない。

列のコピーは `ChatMessage.__table__.columns` を回して行う（列追加に自動追従）。
`ChatMessage` に列を足したら `ChatMessageVariant` にも同じ列を足すこと
（足し忘れた列は退避時に落ちる）。

### 枝の並びと現在位置

同じ `anchor_message_id` を持つ退避グループ＋本線にいる応答を、
**各グループ先頭行の `created_at` 昇順**で並べる。戻しても `created_at` は
元の値のままなので、往復しても番号は変わらない。

### 退避・復元

- **退避（引き直し確定時）**: 旧応答を `chat_messages` から消し、同じ中身を
  `chat_message_variants` へ移す。封筒は retract（削除時と同じ）、LanceDB
  `chat_turns` からも消す（選ばれなかった応答が想起に出ないように）。
- **復元（◀ ▶）**: 本線の応答を退避し、選んだグループを `chat_messages` へ戻す。
  封筒は retract を取り消さず**新しい `chat.message` 封筒を append** する
  （シナリオ枝の再活性化と同じ。めぐりの不可逆性を保つ）。LanceDB へ再登録する。

### 退避しないケース

- 引き直しの起点が最新のユーザ発話でない（置き換え対象にユーザ発話が含まれる）とき。
  過去の応答からの引き直しは下流ごと巻き戻る操作なので、従来どおり物理削除する。
- ユーザ発話の**編集**・**削除**（`delete_chat_messages_from`）。発話が変わる以上
  それに対して引いた応答は無効なので、その発話を起点とする退避グループも消す。
- セッション削除時は退避グループも消す。

### API

| メソッド | パス | 内容 |
|---------|------|------|
| GET | `/api/chat/sessions/{sid}` | 最新ターンのキャラ応答に `variant_index` / `variant_count` を付ける |
| POST | `/api/chat/sessions/{sid}/messages/stream` | `done` の `character_message` に同上 |
| POST | `/api/chat/sessions/{sid}/variants/activate` | body `{anchor_message_id, index}`（1 始まり）。切替後の最新ターン応答群を返す |

### UI

`MessageList` が末尾のキャラ応答にだけ `variantIndex` / `variantCount` を渡し、
既存の `MessageActionBar` の `VariantNav` をそのまま使う。

## 不採用にした案

### シナリオと同じ `is_active` フラグ方式

シナリオは履歴の読み取りが `list_scenario_turns` 1か所に集約されていたので、
フィルタを1つ足すだけで済んだ。1on1 の `chat_messages` は `chat_store` /
`instrument_store` / `timeline_store` / `chronicle_job` / `character_store` /
`external_scenes`（うつつへの流し込み）/ gate まわりなど数十か所が直接問い合わせており、
1か所漏らすだけで選ばれなかった応答がクロニクルやうつつへ混ざる。別テーブルにすれば
既存の読み取り側は一切触らずに済むため、こちらを採った。

### 過去ターンの応答も切り替えられるようにする

下流の巻き戻しと復元（シナリオの案2.5相当）が要る。1on1 では需要が薄いので見送り。

### 同じ行の `content` 等を入れ替える（ID を固定する）方式

ID が変わらないので LanceDB の upsert は楽だが、`log_message_id` や `reasoning` など
入れ替える列を個別に列挙することになり、列追加への追従が弱い。行ごと移す方式にした。

## 対象外（既知の限界）

- 選ばれなかった応答の中でキャラクターが行ったツール操作（`inscribe_memory` や
  WM への投稿など）は残る。従来の引き直しでも同じで、今回は扱わない。
