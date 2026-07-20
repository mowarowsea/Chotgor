# speak_later 仕様書 — キャラ発の時限発話（発話予約）

> Status: **implemented**（2026-07-20 要件合意 → 同日実装完了。実装時の補足判断は「実装メモ」参照）
> 関連: [aliveness_plan.md](aliveness_plan.md)（能動配達・行動権 push）、
> [schedule_plan.md](schedule_plan.md)（availability・生活カレンダー）

## 概要

キャラクターが 1on1 の会話中に「この時刻に、自分からこの会話に声をかける」という
**発話予約**を仕掛け、時刻が来たら Chotgor がキャラクター本人へ問い合わせて、
その会話にキャラ発のメッセージが増える機能。

「予約」はあくまで黒子（システム）側の語彙。キャラクター本人に見せる文言では
「自分がやろうとしていること（能動的な段取り）」として表現し、「予約」とは言わない。

- ① キャラがツール `speak_later` で「何時何分に声をかける」を仕掛ける
- ② 毎分スケジューラが時刻到来を検知し、1on1 同等のヘッドレス生成で発話を落とす
- ユーザには ntfy 通知が飛ぶ（既存 `notify_character_spoke`）。画面はリロードで反映

使用例: 「21時になったら結果教えるね」「夜にもう一回声かける」— 会話の中で
本人が言った「またあとで」を、現実の時刻で本当に実行するための機構。

## 命名について

- **ツール名**: `speak_later` — 本人視点で意味が最も直截（あとで自分から話す）
- **和名は付けない** — 和名は機能群（うつつ・めぐり・なりゆき）に付ける流儀。
  個々のツールには付けない（他ツールと同様 — ユーザ裁定 2026-07-20）
- **内部命名は「予約（reservation）」でよい** — 「予約」を避ける制約はキャラクターに
  見せる文言だけ（同裁定）。テーブル `speech_reservations`・時刻カラム `speak_at`

### 棄却した命名案（検討記録）
| 案 | 理由 |
|---|---|
| 和名「ころあい」等 | 個々のツールに和名は付けない方針（機能群に育ったら再検討） |
| キャラ向け文言での「予約」 | 外部サービス感が強い（ユーザ裁定）。※内部命名・黒子側の表現は対象外 |
| 「約束」 | 相手との約束とは限らない（自分の段取りの場合もある） |
| `plan_return` | 「戻る」に限定される。同席中の数時間後にも使うので不正確 |
| `set_intention` | めぐりの意図（intents）と語彙衝突する |

## 要件（2026-07-20 合意）

1. キャラが任意のタイミングで**現在のチャットセッション**への時限発話を仕掛けられる
2. 仕掛けは ANTICIPATE のようなタグではなく **MCP/tool-use ツール**
3. 発話内容は仕掛け時に固定せず、**発火時に本人が生成**する（仕掛けに残すのは時刻＋用件メモ）
4. 指定時刻が offline（availability 不可）なら仕掛け時に**エラーを返す**
   （availability の上書きはしない。可能なら「予定を変えれば置けるよ」のヒントを添える）
5. キャラに見せる文言は「予約していた件」ではなく**「これをやろうとしていました」**の系統
   （制約はキャラ向け文言のみ。システム内部・黒子側は「予約」でよい — 2026-07-20 裁定）
6. **キャラ設定画面でオン・オフできる**（ユーザが意図しないリクエスト＝課金を発生させる機能のため、既定 OFF）
7. データは**新規テーブル**（intents 相乗りしない）
8. フロント反映は push 不要。通知（ntfy）→ ユーザがリロード、で v1 は足りる
9. 予約が入っている間は、キャラへの問い合わせプロンプトに
   「{時刻}になったら『{note}』をしようと思っている」という一文を注入する（2026-07-20 追加）

## 設計

### 全体フロー

```
1on1 ターン中に本人が speak_later(at, note) を呼ぶ
  → バリデーション（機能有効 / 時刻解釈 / 未来 / 上限 horizon / 未来 availability）
  → speech_reservations に pending 行（同一セッションの既存 pending は superseded に倒して置き直し）
毎分スケジューラ（main.py _run_every_minute 新規登録 name="speech_reservation"）
  → pending 走査: speak_at 到来 && availability.available && 日次 cap 未達 → 発火
発火 = 能動配達（services/gate/delivery.py）と同じヘッドレス SceneLoop:
  build_1on1_chat_request → OneOnOneRouter/Executor → キャラ発メッセージ保存
  → index → ntfy 通知 → scheduler_decisions 記録 → status=fired
```

### ① ツール `speak_later`

- **露出文脈**: 1on1 専用（`origin=="real"` かつ `session_id` あり）かつ
  キャラの機能トグル ON。`context_tools.py` の `CONTEXT_TOOL_SPECS` へ追加
  （reach_out / visit_user / override_schedule と同列）。hint も同所から注入
- **引数スキーマ**:
  - `at` (string, required): `"HH:MM"`（次にその時刻が来る時点 — 今日 or 明日）
    または `"YYYY-MM-DD HH:MM"`。上限 horizon **72時間先まで**（超過はエラー）
  - `note` (string, required): 何を話そうとしているかの短いメモ（本人の言葉。
    発火時に本人へそのまま返る）
- **バリデーション（実行側）**:
  1. トグル OFF / origin 不一致 / session なし → エラー文字列（露出とのタイムラグ対策。
     messenger.py と同じ二重ガード）
  2. 時刻が過去・パース不能・72時間超 → エラー文字列
  3. `check_availability(char, speak_at, sqlite=...)` で `available == False`
     → エラー文字列「その時間は{reason}なので声をかけられない」。
     生活カレンダー有効キャラには「予定を動かせば置ける（override_schedule）」のヒントを添える。
     ※未来時刻評価は予報パネルと同じ**無風仮定**（うつつシーン進行中などの
     実行時状態は False 扱いで評価する）
  4. 同一セッションに既存 pending → **置き直し**（旧行を status=superseded、新行を作成。
     「やっぱり22時にする」を自然に許す。履歴は行として残る）
- **ツール結果文言**（「予約」を使わない）:
  「{時刻}に自分から声をかける心づもりを置いた。時間が来たらこの会話に戻ってくる」

### ② スケジューラと発火

- `main.py` に `_run_every_minute(name="speech_reservation", ...)` を新規登録
  （A1 共通ループ — heartbeat `scheduler_heartbeat_speech_reservation` が自動で立つ）
- 発火判定（毎分・LLM 呼び出しは発火時のみ）:
  1. `speak_at <= now` の pending を走査
  2. `check_availability` を再評価。unavailable なら**発火せず pending 維持**
     （置いたあとに予定が変わった／うつつシーン進行中のケース。復帰後に遅れて発火）
  3. `speak_at + 24h` を過ぎても発火できなければ **status=expired**
     ＋ scheduler_decisions（declined）。黙って消さず記録に残す
  4. 日次コストガード: `escrow_delivery_daily_cap` の予算・カウンタを**共有**
     （キャラ発の現実接触は経路を問わず1つの予算 — 2026-07-11 裁定の延長）。
     到達日は skipped 記録（日1回）→ 翌日カウンタリセット後に遅延発火
  5. セッション削除済み・キャラ estranged・退席済み → status=cancelled ＋記録
- 決定ログ: `scheduler_decisions` kind=`speech_reservation`（fired / declined / skipped / error）

### 発火時の生成（キャラクター問い合わせ原則）

能動配達 `_deliver_session` と同じ 1on1 SSE 等価経路（履歴・WM ブロック込み）で生成する。
違いは「配達すべきユーザメッセージが無い」こと。**合成注釈**を最終ユーザターン相当として
LLM に渡す（**DB には保存しない** — 時間差注釈と同じ「LLM 渡しのコピーのみ」思想。
画面にはキャラの発話だけが増える）:

- 定刻発火: 「（いま {HH:MM}。あなたはこの時間に『{note}』をやろうとしていた。
  ここからはあなたから声をかける番）」
- 遅延発火: 「（いま {HH:MM}。本当は {speak_at} に『{note}』をやろうとしていたが、
  都合がつかず今になった）」

**未配達メッセージとの合流**: 発火時にセッションへ未配達のユーザメッセージが残っていたら、
escrow 配達と同じ手順（時間差注釈＋`mark_messages_delivered`）で併せて配達し、
合成注釈を末尾に添える。別々に2ターン発生させない。
実装は `_deliver_session` に「追加注釈・ユーザメッセージ無しでも生成する」拡張を入れて共用する。

### 予約中のプロンプト注入（自分の段取りの想起）

pending の予約がある間、キャラへの問い合わせプロンプトに本人の段取りとして一文を注入する:

> {speak_at} になったら「{note}」をしようと思っている。

- 注入先は予定コンテキスト（`services/schedule/awareness.build_schedule_lines` →
  ターン注釈 `{block_schedule}`）と同じ枠に相乗りする（1on1・うつつ PC の両方に載る）
- 目的: 本人が自分の段取りを覚えている状態を作る。発火前に会話が動いたときの
  置き直し判断、「もうすぐ◯時だ」といった自然な言及、発火時の唐突さの緩和
- 予約はセッションに紐づくが、注入はセッションを問わない
  （自分の段取りはどの場でも本人の頭の中にあるものなので）
- fired / expired / cancelled / superseded になったら注入は消える

### コストガードとトグル

| ガード | 内容 |
|---|---|
| キャラ単位トグル | `characters.speak_later_enabled`（INTEGER, 既定 0=OFF）。キャラ編集 UI にチェックボックス。OFF ならツール非露出＋実行ガード |
| 日次 cap | `escrow_delivery_daily_cap` 共有（発火時に消費。仕掛け自体はコストゼロなので消費しない） |
| pending 上限 | 1件/セッション（置き直しで上書き）。キャラ全体の上限は設けない（セッション数と日次 cap が実質上限） |
| horizon | 72時間先まで |

### データ

新規テーブル `speech_reservations`（migrations.py にマーカー冪等のマイグレーション追加）:

```sql
speech_reservations
  id            TEXT PRIMARY KEY
  character_id  TEXT NOT NULL
  session_id    TEXT NOT NULL
  speak_at      DATETIME NOT NULL   -- 発火予定時刻
  note          TEXT NOT NULL       -- 何を話そうとしているか（本人の言葉）
  status        TEXT NOT NULL       -- pending / fired / expired / cancelled / superseded
  created_at    DATETIME NOT NULL
  fired_at      DATETIME NULL
```

`characters` へ `speak_later_enabled INTEGER DEFAULT 0` を追加。

タイムライン封筒: 発火で保存されるキャラ発話は通常の `chat.message`（dual-write で
封筒が自動的に載る）。仕掛け行為自体は `tool_call_events`（ToolExecutor 集約）と
scheduler_decisions に残るため、**専用の封筒 event_type は v1 では作らない**（将来枠）。

## 改修ファイル見積もり

| 区分 | ファイル | 内容 |
|---|---|---|
| 新規 | `backend/character_actions/later_speaker.py` | SCHEMA / DESCRIPTION / HINT ＋ ツール実装（バリデーション・予約書き込み） |
| 新規 | `backend/services/gate/speech_reservation.py` | 発火ランナー（毎分走査・発火判定・生成呼び出し）。delivery.py の隣 |
| 変更 | `character_actions/tool_specs.py` | CONTEXT_TOOL_SPECS へ1件追加 |
| 変更 | `character_actions/context_tools.py` | 露出判定（real＋session＋トグル） |
| 変更 | `character_actions/executor.py` | `_dispatch` に分岐追加 |
| 変更 | `services/gate/delivery.py` | `_deliver_session` の共用化（追加注釈・pending 無し生成） |
| 変更 | `main.py` | `_run_every_minute(name="speech_reservation")` 登録 |
| 変更 | `services/schedule/awareness.py`（＋flow.py の注釈組み立て） | 予約中の一文をターン注釈へ追加 |
| 変更 | SQLiteStore / migrations.py | テーブル＋CRUD＋characters 列 |
| 変更 | `templates/character_edit.html` ＋ characters API | トグル UI・保存 |
| 変更 | WorkDir `.claude/settings.json` | permissions.allow へ `mcp__chotgor__speak_later` |
| 変更 | `docs/current-spec/ARCHITECTURE.md` | 機構の追記 |

## 実装メモ（2026-07-20 実装時の補足判断）

仕様に書かれていなかった細部は以下のとおり決めた:

| 論点 | 判断 |
|---|---|
| `at` の 24時超え表記 | override_schedule と同じ流儀で `"25:30"` = 翌1:30 も受ける（`parse_speak_at`。本人が override の習慣で書いても弾かない） |
| 遅延発火の文言切替 | speak_at から **10分以上** 遅れたら遅延文言（毎分スケジューラの粒度では数分の遅れは定刻扱い） |
| 発火側のトグル再確認 | 仕掛け後にユーザがトグル OFF にしたら発火させず **cancelled** に倒す（トグルは課金ガードなので発火側でも見る — 仕様の発火判定リストへの追加） |
| fired マークの順序 | fired 遷移＋日次カウンタ消費を**生成より先に**確定する（生成失敗で毎分 LLM を叩き直さない。escrow の delivered-before-LLM と同思想。失敗は決定ログ error に残る） |
| cancelled 系の決定ログ | outcome=**skipped**（物理で流れた扱い。expired のみ仕様どおり declined） |
| 注入ブロックの見出し | `{block_schedule}` の見出しを「## あなたの予定（生活カレンダーより）」→「## あなたの予定」へ中立化（カレンダー無効キャラにも予約行が載るため。スナップショット更新済み） |
| escrow ready マーカー | 発火で未配達分を併せて配達したら `escrow_ready_{session}` を掃除する（stale な復帰観測時刻が次回配達のジッターを飛ばさないように） |

テスト: `tests/test_later_speaker.py`（仕掛け側）・`tests/test_speech_reservation.py`（発火側）・
`tests/test_context_tools.py` / `tests/test_schedule_awareness.py`（露出・注入の追記分）。

## 棄却した設計案（検討記録）

| 案 | 棄却理由 |
|---|---|
| タグ方式（`[ANTICIPATE_RESPONSE]` 型） | ANTICIPATE は副作用なしの独白。本件は状態を作る行為なので、inscribe 等と同じ「本人の行為＝ツール」に揃える。構造化引数・将来のキャンセル/一覧拡張もツールが素直 |
| 発話本文を仕掛け時に固定 | 発火までに起きたこと（ユーザの追加発言等）を無視した発話になる。発火時生成なら本人が状況を見て言葉を選べる（キャラクター問い合わせ原則とも整合） |
| `intents` テーブル相乗り | 意図は圧力導出で「いつか」拾われる経済。本件は「何時何分」の決定論で性質が異なる（ユーザ裁定・2026-07-20） |
| offline 時刻でも仕掛け可（availability 上書き） | 就寝中に発話→返信したら「寝てます」になる矛盾。仕掛け時エラーで返す（ユーザ裁定） |
| 新規セッションへ発火（push 型） | 要件は「現在のチャット」への発話。時限 push（reach_out の時刻指定版）は将来枠 |
| フロントへのリアルタイム push（SSE/WS） | 既存 UI に push 基盤がなく新設コストが大きい。ntfy 通知→リロードで v1 は足りる（ユーザ裁定）。将来枠 |
| cap 到達日はツール非露出（reach_out 方式） | 仕掛け自体はコストゼロで、コストは発火時。露出は絞らず発火側で cap を見る |
| 発火時のユーザ会話中ガード | escrow 能動配達も同種のレースを許容している。最悪でもターンが交錯するだけで、決定ログに残る。v1 では作らない |

## 将来枠

- 予報パネル週間カレンダーへの予約表示（伏せ枠ネタバレ制御と同じ扱いの検討が必要 —
  note をユーザに見せるかは本人の秘密性の論点）
- キャンセル・一覧ツール（`speak_later` の置き直しで v1 は代替）
- 時限 push（セッション外・reach_out の時刻指定版）
- フロントへのリアルタイム push 基盤

## テスト観点

- バリデーション純関数（時刻解釈・horizon・過去時刻・offline エラー・ヒント付与）
- 置き直し（superseded 遷移・pending 一意性）
- 発火判定（定刻・availability 延期→遅延発火・24h expired・cap skipped・翌日遅延発火）
- セッション削除／estranged／退席済みの cancelled 遷移
- 未配達メッセージとの合流（1ターンに併合されること・mark_messages_delivered 順序）
- 合成注釈が DB に保存されないこと（画面にはキャラ発話のみ増える）
- 予約中のプロンプト注入（pending 中のみ載り、fired/expired/superseded で消えること）
- トグル OFF 時の非露出＋実行ガード
