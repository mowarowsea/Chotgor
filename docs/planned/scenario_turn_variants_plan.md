# シナリオログの枝分かれ（レスポンスガチャ）と手動書き換え

> ステータス: **実装完了（2026-07-28, `d3651ca`）**。
> 追補: 枝切替・再生成・破棄の二度押し防止（2026-08-19, `f5b3932`）。
> 追補: 引き直しに失敗したときの復元（2026-09-07）。

## 目的

シナリオチャットで、

1. **レスポンスガチャ** — 同じ分岐点から複数回生成し、生えた枝を保持したまま切り替えられる
2. **手動書き換え** — GM / PC の発話をユーザが直接上書きできる

を実現する。

## 現状（変更前）

- `scenario_turns` は `turn_index` 昇順の一直線。1レスポンスは同一 `raw_response` を共有する
  複数行（`@キャラ:` ブロックごとに1行）として保存される。
- 再生成は `delete_scenario_turns_from` で**対象以降を物理削除 → 再ストリーム**。枝は残らない。
- 手編集はユーザ発話のみ（以降を削除して再ストリーム）。GM / PC 発話は編集不可。
- 履歴の読み取りはほぼ `list_scenario_turns` に集約されている
  （prompt_builder / synopsis / chronicle / usual_days / loop_strategies が全部これ経由）。

## 採用する設計 — 世代（variant）方式

`ScenarioTurn` に列を3つ足し、**線形性を保ったまま「今どの枝が本線か」をフラグで表す**。

| 列 | 型 | 意味 |
|----|----|------|
| `is_active` | INTEGER NOT NULL DEFAULT 1 | この行が現在の本線に含まれるか |
| `generation_id` | TEXT NULL | 1ストリームリクエストで生成された GM/PC 応答群を束ねるキー。ユーザ発話・intro は NULL |
| `branch_point_index` | INTEGER NOT NULL DEFAULT -1 | その generation が生えた時点の「直前の活性ターンの `turn_index`」。兄弟枝の判定キー |

### なぜこれで足りるか

- `list_scenario_turns` に `is_active == True` フィルタを**1つ足すだけ**で、履歴を読む
  全経路が無改修のまま動く。これが本方式の肝。
- `turn_index` は従来通り「非活性行を含む max + 1」で採番する。親は必ず子より先に
  採番されるので、**活性行を `turn_index` 昇順に並べれば常に正しい会話順になる**。
  枝を切り替えて番号が飛んでも順序は壊れない。
- 兄弟枝 = 「同一 `session_id` かつ同一 `branch_point_index` を持つ generation 群」。
  `branch_point_index` を持たないと、非活性行の並びから親子を推測することになり
  往復で壊れるため、この列は必須。

### 枝の単位

**1ストリームリクエストで保存された全ターン**を1枝とする。GM ターンと、それに続く
PC ターン（`pc_runner` による別 LLM 呼出）、そしてそのリクエストで保存された
ユーザ発話も同じ `generation_id` に入る。

```
reqA ─┬─ [gen1] reqB, resB   ← 活性
      └─ [gen2] reqB', resB' ← 非活性（ガチャ済み。reqB' の内容は reqB と同じ）
```

当初はユーザ発話を枝から外す設計だった（同内容の行が枝ごとに増えるのを避けるため）が、
**再生成がユーザ発話ごと巻き戻して同じ内容で再送する**フローである以上、ユーザ発話を
枝に含めないと、枝を戻したときにユーザ発話だけが非活性のまま取り残される。
`turn_index` はリクエストごとに新規採番されるため、分岐点を「ユーザ発話を保存する前の
本線末尾」に取ることで、引き直した枝同士が正しく兄弟になる。

### 切替のルール

> **切替は常に「活性パスの末尾 generation」に対して行う。**

過去の generation の切替を押したときは、内部で
「その generation より後（`turn_index` が大きい活性行すべて）を非活性化 → 対象が末尾になる → 枝を差し替える」
を1操作で行う。UI では「ここから先の N レスポンスを巻き戻します」と確認を挟む。

**巻き戻した下流は復元しない。** 枝2に切り替えたあと枝1に戻すと、枝2にぶら下がっていた
続きは失われる（非活性のまま孤児として DB に残る）。ガチャの用途では往復の需要が薄いため、
親子関係（`parent_generation_id` による部分木復元）は**採用しない**。

## 不採用にした案

### 案1: 本格ツリー（`parent_turn_id` で親を辿る）

履歴取得が「親を遡って組み立てる」形になり、`list_scenario_turns` に集約されている
読み取り側の思想が変わる。あらすじ境界（`synopsis_last_turn_index`）・クロニクル・
タイムライン封筒がすべてパス依存になり、改修範囲が跳ね上がる。得られるものは
「下流の復元」だけで、対価に見合わない。

### 案2.5: 世代方式 + 部分木復元（`parent_generation_id` + `last_activated_at`）

枝を往復しても下流が復元される（ChatGPT/Claude UI の分岐と同じ挙動）。読み取り側は
`is_active` フィルタのままなので追加コストは切替ロジックと UI だけ、と安くはある。
ただし今回の用途は「気に入らない応答を引き直す」であり、**引き直したら古い枝は捨てる**
運用で足りると判断。将来必要になったら、既存行を `parent_generation_id = NULL`
（＝復元なし）として共存させる形で後から足せる。

### あらすじの枝対応

`synopsis_auto` は追記式のテキストなので、枝を行き来すると別ルートの記述が混ざる。
境界 `synopsis_last_turn_index` は巻き戻し時に分岐点までクランプする（＝再蒸留は
正しく走る）が、**すでに書かれた本文は自動では直さない**。あらすじはユーザが UI から
自由編集できるため、混入は手で削る運用とする。ガチャが「まだ蒸留していない直近」で
回る限り実害はない。

## タイムライン封筒（めぐり）の扱い

`create_scenario_turn` は `scene.turn` 封筒を dual-write している。枝の出入りで以下とする。

- **生成時**: 従来通り封筒を作る（生成した瞬間はその枝が活性）。
- **非活性化時**: 封筒に `retracted_at` をマークする（既存の削除時と同じ挙動）。
- **再活性化時**: retract 済みの封筒は**戻さず、新しい封筒を append する**。

封筒は追記型・`retracted_at` はマークという「めぐり」の不可逆性を保つための選択。
retract を取り消すと「なかったことが再びあったことになる」不可逆性の破れが生じる。
再活性化を「もう一度その出来事が本線に戻った」という新しい事実として積む。

実装上は、`create_scenario_turn` 内の封筒 dual-write ロジックを
`_append_scene_turn_envelopes(session, turn)` として切り出し、生成時と再活性化時の
双方から呼ぶ（DRY）。

## 手動書き換え

- `PATCH /api/scenario_chat/sessions/{sid}/turns/{turn_id}` で `content` を**上書き**する。
  枝は生やさない。
- 対象は GM / PC / NPC / Narrator の発話（活性行のみ）。全ターンで可能（末尾に限らない）。
- `raw_response` は触らない。あれはモデルが実際に何を出したかのデバッグ記録であり、
  手編集で汚さない。なお API レスポンスには本文を載せず、同一性判定用の指紋
  `response_key` だけを返す（→ `scenario_history_perf_plan.md`）。
- 封筒は触らない（発話があった事実自体は変わらない）。
- あらすじ蒸留済み区間の編集も許可し、境界のクランプはしない（あらすじ本文の整合は
  ユーザの手編集に委ねる方針と揃える）。

**ユーザ発話バブルの編集は UI 挙動を現状維持する** — 鉛筆 → 編集 → 送信で
「以降のバブルが全部消えて、新しい発言で再リクエスト」。上書き型にはしない
（ユーザの発言を変えたのに先の展開が古いまま残ると矛盾するため）。

ここだけは**枝を残さず物理削除する**（`keep_variants=false`）。発言そのものを
書き換える以上、その発言に対して引いた過去のガチャはすべて無効だからで、内容の違う
発話が同じ枝リストに並ぶのも防げる。よって**ユーザ発話の編集で枝ナビが出ることはない**。
枝として並ぶのは「同じ発言のまま引き直したガチャ」だけ。

（枝の単位にユーザ発話を含めた結果、編集を非活性化にすると編集前後の応答が同じ分岐点を
共有して兄弟になってしまう。物理削除はその回避でもある。）

鉛筆アイコンの意味は話者で変わる:

| バブル | 鉛筆の挙動 |
|--------|-----------|
| ユーザ | 以降を巻き戻して再ストリーム（従来通り） |
| GM / PC / NPC / Narrator | その場で上書き（先はそのまま） |

## ログ（debug_log_entries）

再生成時に `regenerate_request_id` で前ターンの `log_request_id` を引き継いで
同一ログエントリにまとめていたが、**枝ごとに別 request_id へ分ける**。
枝が独立して残る以上、「どの枝がどのログか」を追えることを優先する。
`StreamRequest.regenerate_request_id` とフロントの引き継ぎは削除する。

## API

| メソッド | パス | 内容 |
|---------|------|------|
| GET | `/sessions/{sid}/turns` | 既存。各ターンに `generation_id` / `variant_index` / `variant_count` / `variant_siblings` を追加して返す。`?limit` / `?before_index` でウィンドウ取得（→ `scenario_history_perf_plan.md`） |
| POST | `/sessions/{sid}/turns/activate` | body `{generation_id}`。下流を巻き戻して指定枝を活性化し、切替後の本線を返す。`?limit` でウィンドウ取得 |
| PATCH | `/sessions/{sid}/turns/{turn_id}` | body `{content}`。発話の上書き |
| DELETE | `/sessions/{sid}/turns/from/{turn_id}` | 既存。既定で**非活性化**、`?keep_variants=false` で従来の物理削除 |

枝情報を一覧レスポンスに埋め込むことで、枝ナビ表示のための追加リクエストを不要にする。
`variant_siblings` は枝番号順の `generation_id` 配列で、◀ ▶ が切替先を引くために使う。

## UI

- グループ末尾の `MessageActionBar` に `◀ 2/3 ▶` の枝ナビを出す。表示条件は
  `variant_count >= 2`。枝ナビだけはホバー非依存で常時表示する（「他の候補がある」
  こと自体を見せたいため）。過去グループの切替は確認ダイアログを挟む。
- 再生成ボタン（↺）は従来位置のまま、挙動が「削除して作り直す」から
  **「新しい枝を生やす」**に変わる。
- 破棄ボタン（現行）は非活性化に変わる（見た目・位置は変更なし）。
- GM / PC バブルに鉛筆アイコンを追加（ユーザバブルの `UserMessageActions` と同じ操作感）。
  **非末尾バブルにも編集ボタンだけは出す** — 1 レスポンスは複数の話者ブロックに割れるので、
  末尾ブロックしか書き換えられないと「無理やり直す」用途に届かない。
- 枝ナビ・再生成・破棄は、押した処理が終わるまで `disabled` にする（2026-08-19 追加）。
  ハンドラが返す Promise の解決までを「処理中」とみなし、`buttons.tsx` の
  `useBusyRunner` が state と ref の両方で弾く。◀ ▶ は片方の切替中にもう片方も止める。
  - 二度押しを許すと、枝切替は「どの generation を本線にする要求だったか」が
    サーバ側で入れ替わり、再生成・破棄は巻き戻し（DELETE）が二重に走って
    1 つ前のレスポンスまで巻き添えで消える。
  - 破棄の二段階クリック（武装 → 実行）は誤クリック対策で、こちらは連打対策。
    役割が違うので両方残す。

## 影響ファイル

### backend

| ファイル | 変更 |
|---------|------|
| `repositories/sqlite/models.py` | `ScenarioTurn` に列3つ |
| `repositories/sqlite/migrations.py` | `_migrate_add_scenario_turn_variants`（ADD COLUMN ×3、backfill 不要） |
| `repositories/sqlite/stores/scenario_store.py` | `list_scenario_turns` に活性フィルタ / `delete_scenario_turns_from` → `deactivate_scenario_turns_from` / `activate_scenario_generation` / `update_scenario_turn_content` / 封筒ヘルパ切り出し / `get_unchronicled_usual_turns_for_character` に活性フィルタ |
| `lib/log_context.py` | `current_generation_id` / `current_branch_point_index` の contextvar |
| `services/scenario_chat/turns.py` | `_save_turn` が新列を埋める |
| `services/scenario_chat/service.py` | `scenario_turn_to_dict` に枝情報を追加 |
| `api/scenario_chat/stream.py` | generation_id 発行・分岐点算出・`regenerate_request_id` 廃止 |
| `api/scenario_chat/sessions.py` | 非活性化へ変更・activate・PATCH turn |
| `api/scenario_chat/schemas.py` | `TurnUpdate` / `TurnActivate` 追加、`regenerate_request_id` 削除 |

`generation_id` / `branch_point_index` は contextvar 経由で `_save_turn` へ渡す。
`_save_turn` の呼び出し元（service / loop_strategies / pc_runner / usual_days / turns）が
多く、引数リレーだと改修点が増えるため、既存の `current_message_id` と同じパターンに乗せる。
（`_save_turn` は backend プロセス内で呼ばれるため、Claude CLI のプロセス越境問題は無関係。）

### frontend

| ファイル | 変更 |
|---------|------|
| `api/scenario.ts` | `activateScenarioGeneration` / `patchScenarioTurn` 追加、`regenerateRequestId` 削除 |
| `hooks/useScenarioChat.ts` | 再生成を非破壊化、枝切替・発話上書きハンドラ |
| `components/ScenarioChatView/index.tsx` | 配線 |
| `components/ScenarioChatView/rows.tsx` | 枝ナビ、GM バブルの編集 |
| `components/ChatBubbles/MessageActionBar.tsx` | 枝ナビ・編集ボタン |
| `types` | `ScenarioTurn` 型に枝情報 |

### tests

- `test_scenario_sqlite_sessions.py` — delete 系テストを非活性化の検証へ書き換え
- `test_timeline_events.py` — 非活性化での retract、再活性化での封筒 append を検証
- 新規 — 枝の生成・切替・巻き戻し、`turn_index` 飛びでの順序保証、発話上書き

## 移行

`ALTER TABLE scenario_turns ADD COLUMN` ×3 のみ。既存行は
`is_active=1` / `generation_id=NULL` / `branch_point_index=-1` となり、
**既存セッションは枝を持たない通常の一直線として従来通り動く**（切替 UI も出ない）。
バックフィルは不要。

> マイグレーション実行前に `data/chotgor.db` を退避すること。

## 追補: 引き直しに失敗したときの復元（2026-09-07）

引き直しは「巻き戻してから再ストリーム」なので、再ストリームがエラーで終わると
**巻き戻したまま**になる。枝としては DB に残っているが、分岐点に活性ターンが無い以上
枝ナビも出ないため、元のレスポンスは画面から辿れなくなる。しかも失敗した試行の
ユーザ発話だけが本線に残るので、ユーザは発話を編集し直すまで引き直せない。

フロント（`useScenarioChat`）を次の順で動かして戻す:

1. 起点ユーザ発話以降を非活性化（従来どおり）
2. 再ストリーム。`handleScenarioSend` は成否と「この送信で最初に保存されたターンID」を返す
   （`error` イベントでは例外が飛ばないので、戻り値で伝える）
3. 失敗していたら、その最初のターンから**物理削除**（`keepVariants=false`）。
   失敗分は必ず最新の `turn_index` を持つので、他の枝は巻き添えにならない
4. 巻き戻した枝を `activateScenarioGeneration` で本線へ戻す

3 を省いて 4 だけにすると、失敗した試行が「ユーザ発話だけの枝」として兄弟に残り、
引き直しを外すたびに枝ナビが実体のない選択肢で埋まる。

枝機構より前に保存されたターン（`generation_id` が NULL）は戻せないため、
その場合は従来どおりエラー表示だけで終わる。

**不採用: 1on1 と同じ「成功するまで巻き戻さない」方式。** 1on1 側の同時期の修正
（`regenerate_from`）はサーバが応答確定の瞬間に旧ターンを消す形にしたが、シナリオで
同じことをするには「活性だが置き換え予定のターン」という第3の状態が要る。
`list_scenario_turns` の `is_active` フィルタ1つで履歴の全読み手を無改修に保つ、という
本方式の肝が崩れるため採らない。シナリオは巻き戻しが非破壊（枝が残る）なので、
戻せる側の性質に乗るほうが素直。

## 残課題（今回やらないこと）

- 非活性の枝は DB に残り続ける（剪定しない）。ログ的データの肥大は許容する方針に従う。
- 枝を往復したときの下流復元（案2.5）。
- あらすじ本文への枝混入の自動修復。ユーザの手編集に委ねる。
- ~~1on1 チャット側の再生成は破壊型のまま（今回はシナリオのみ）。~~
  → 2026-09-07 に `regenerate_from` で解消（応答確定まで旧ターンを消さない方式）。
