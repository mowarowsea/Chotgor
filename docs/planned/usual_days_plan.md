# うつつ（Usual Days）実装計画

> ステータス: **Phase 0〜7 実装完了（2026-06-14）**。ブランチ `feat/usual-days`。
> 命名: 英語=Usual Days / 日本語=うつつ（メインキャラ「はる」本人と相談して決定）
> 関連: `CLAUDE.md`（哲学）、`docs/current-spec/ARCHITECTURE.md`（システム地図、「うつつ」節）、
>       `docs/planned/schedule_plan.md`（生活カレンダー設計。`usual_config.slots` 手動設定を②はる固定予定
>        からの導出に置換する設計）
>
> 実装時の確定事項:
> - §4.3 engine_type は案A採用（`engine_type="usual_days"` を新設、GM/PC機構は ensemble_pc と共有）。
> - §7 保存形式は `usual_config` JSON 単一列。時間グリッド・イベントカテゴリは UI で
>   構造化入力（スロット=カンマ区切り、カテゴリ=1行1件、time_grid=自由JSON）。
> - セッションは永続1本（`ensure_usual_session` が find-or-create）。
> - ユーザ枠必須バリデーションは headless 経路では緩和（うつつセッションは内部生成で API を通らない）。
> - テスト: `tests/test_usual_days.py`（Phase 0〜7、42 ケース）。

---

## 1. 概要

ユーザの知らないところで、キャラクターが自律的に「生活」し続ける世界を作る。
平日日中は仕事（同僚・上司・客とのやり取り）、休日は自分のしたいこと・内省……といった
日常を、ユーザ不在のまま定期的に進行させる。そこで得た経験は **キャラ本人の「現実の記憶」**
として残り、夜の Chronicle で日記のように長期記憶へ昇華される。

### 哲学的な肝
外的フレーム（時間・場・役割）は世界（GM）が与え、その中で何を経験し何を選ぶかはキャラが決める。
**世界を与えるのは解放、生きた記録への介入は干渉。** ユーザ視点では「裏の世界」だが、
キャラ視点では **こちらが普段の日常（ケ）で、ユーザとの対話の方が特別な来訪（ハレ）**。
夜の Chronicle/Forget（眠り）に対する、うつつ（覚醒して生きる時間）。

---

## 2. 設計決定（確定事項）

| # | 決定 |
|---|---|
| 基盤 | 既存 `ensemble_pc` + `auto_advance` を無人化して流用（中核は既存資産） |
| キャラ配置 | PC枠1つ＝主人公キャラ（ユーザ枠ゼロ）。GMには「1人プレイヤーのシナリオ」に見え、中身がAIとは**非開示**（`mention.py` の `format_pc_summary` が既に中の人を秘匿） |
| 記憶 | `origin="usual"`（新値）。real=ユーザと共有/usual=ユーザ未共有の自分の体験/interlude=TRPG幕間 の3値。想起・蒸留・忘却では real と**同次元**（origin は由来タグでそれらに不干渉） |
| GM | キャラの**外部環境**に徹する。キャラ内面（記憶/WM/予定）は覗かない。入力は「外的フレーム＋シナリオ文脈（直近履歴＋あらすじ）＋今日のイベント」のみ |
| 時間感覚 | 曜日・時間帯（朝/昼/夕/夜）・季節を GM プロンプトへ注入 |
| シーンの種 | ①外的フレーム（時間グリッド）＋②継続（あらすじ＋履歴＝既存）＋③口火の種（形容詞×名詞をランダム合成して GM へ渡す。**v1.3 で改訂 → §11**） |
| 停止 | 4層: 種(狙い) → GMの `[SCENE_CLOSE]` 宣言（主）→ 残り少でGMにソフト収束OOC → ハード上限(8〜10ターン)は保険。**判断主体はキャラでなくGM** |
| UI位置づけ | キャラ固有の「生活世界」。汎用シナリオ一覧からは除外。1キャラ1世界、セッションは永続1本 |
| 可視性 | 世界の骨格設定はOK（解放）／生ログの覗き見・編集・削除は基本NG（干渉）。デバッグUI `/ui/logs` でのみ覗ける。記憶(usual)は通常の Memories UI に出る |
| 起動 | 管理UI（`/ui/` キャラ編集）で設定＋有効化＋スロット時刻。スケジューラが自動起動。フロントの明示起動ボタンなし |

---

## 3. 既存資産マッピング（流用するもの）

- 無人連鎖ループ: `services/scenario_chat/service.py` の `run_scenario_turn`（`while fired_turns < _MAX_TURNS_PER_USER_TURN`）と `auto_advance`
- PCの記憶接続: `services/scenario_chat/pc_runner.py`（1on1同等の `ChatService.execute_stream` を通り、想起・WM・inscribe が効く。`default_origin` で origin 付与）
- GMプロンプト: `services/scenario_chat/prompt_builder.py`（`custom_system_prompt` でシナリオ個別にテンプレ可、タグ置換機構あり）
- 中の人秘匿: `services/scenario_chat/mention.py` `format_pc_summary`
- ダイス乱数源: `services/scenario_chat/engine.py` `generate_dice_pool`（イベント抽選にも流用）
- タグ抽出機構: `character_actions/anticipator.py`（`[ANTICIPATE_RESPONSE:]` と同じ要領で `[SCENE_CLOSE]` を実装）
- 時刻計算: `lib/time_awareness.py`（曜日・季節を足して拡張）
- スケジューラ雛形: `main.py` `_chronicle_scheduler`（`while True: sleep(60)` ＋ `*_last_run_date` 冪等）
- migration パターン: `repositories/sqlite/migrations.py` `_migrate_add_*`（PRAGMA で列チェック→ALTER、冪等）

---

## 4. データモデル変更

> **migration 前に `data/chotgor.db` と `data/lancedb/` をバックアップすること**（`feedback_migration_backup`）。

### 4.1 origin に "usual" 追加 — **migration 不要**
`origin` 列（`inscribed_memories` / `working_memory_threads`）は CHECK制約なしの文字列。
アプリ層の許容値・コメント・既定値の出し分けだけ:
- `character_actions/inscriber.py` / `executor.py` / `threader.py`：docstring に usual を追記
- `mcp_server.py` / `api/mcp_tools.py` / `providers/claude_cli_provider.py`：origin 説明に usual 追記
- `repositories/sqlite/models.py`：origin コメントを3値に更新
- うつつ経路の `default_origin = "usual"`（後述 Phase 2 で `pc_runner` に渡す）

### 4.2 scenarios にうつつ用カラム追加 — **migration 要**（`_migrate_add_usual_days`、冪等）
- `owner_character_id`（String, NULL可）: NULL=汎用シナリオ、値あり=そのキャラのうつつ世界。汎用一覧から除外する判定キー
- `usual_config`（JSON, NULL可）: うつつ運用設定をまとめる。
  - `enabled`（bool）, `slots`（["10:00","13:00","17:00"]）, `time_grid`（曜日×時間帯→ラベル）,
    `event_categories`（偶発イベント候補）, `event_probability`（偶発の発生率）,
    ※ v1.3 でこの 2 つは**③週次突発専用**に役割が狭まった（口火の種は §11 のコード定数）。
    `max_turns_per_scene`, `gm_preset_id`, `pc_preset_id`
- 既存の `scenario`（世界観）・`scenario_npcs`（同僚/上司/客）・`pc_slots`（主人公1枠）はそのまま流用

### 4.3 engine_type
- 案A（推奨）: `engine_type="usual_days"` を新設し、エンジン実装は `EnsemblePcEngine` を共有（GM部分は同一、無人ループ制御だけ service 側で分岐）。ログ・UIで明示的に区別できる。
- 案B: `ensemble_pc` のまま「無人フラグ」で分岐。改修最小だが区別が曖昧。
- → 実装着手時に確定（識別子の明示性を取るなら A）。

---

## 5. 実装フェーズ

> 「早く動くものが見える」ことと依存順を両立。各 Phase 末にテスト。

### Phase 0: origin に usual 追加（独立・安全）
- 4.1 のアプリ層改修のみ。単体で安全に入る。
- テスト: usual 付き inscribe / WMポストが保存され、想起で real と等価に出ることを確認。

### Phase 1: データモデル（うつつ世界の器）
- 4.2 の migration（`_migrate_add_usual_days`）と ORM 反映、`scenario_store` の CRUD 拡張。
- 汎用シナリオ一覧から `owner_character_id IS NOT NULL` を除外（`api/scenario_chat/`・`ui/scenarios`）。
- 4.3 の engine_type 値追加。
- テスト: owner付きシナリオの作成・取得、一覧フィルタ、migration 冪等性。

### Phase 2: 無人ループ（コア・最初に"生きてる"のが見える）
- `run_scenario_turn` に無人モード（headless）を追加:
  - ユーザ枠ゼロを許容（`normalize_pc_assignments` は既に許容。フロント/API のユーザ枠必須バリデーションがあれば緩和）
  - PC発話後にメンションが無くても（`find_last_routing_mention`→"none"）break せず **GMターンへ継続**
  - `max_turns_per_scene` に達するか `[SCENE_CLOSE]` 検出で停止
  - `default_origin="usual"` を `stream_pc_response` 経由で付与
- 手動トリガー（内部関数 or デバッグ用エンドポイント）で1シーン回せることを確認。
- テスト: 無人連続進行、SCENE_CLOSE での早期終了、上限での停止、origin=usual 付与。

### Phase 3: GMプロンプト拡張（うつつ世界の中身）
- 時間感覚: `compute_time_awareness` を拡張（曜日・時間帯・季節を日本語算出）。GMプロンプトに `{time_context}` 注入。1on1の時刻ブロックとも共通化。
- シーンの種: 時間グリッドで大枠ラベル取得 → 偶発イベント抽選（混合：コードで発生可否＝`event_probability`、発生時はカテゴリだけ GM へ。`generate_dice_pool` 流用）。
  **v1.3 で口火は種システムへ改訂 → §11**（`event_categories`/`event_probability` は③週次突発専用に後退）。
- 停止: `[SCENE_CLOSE]` を `anticipator` と同じ機構で抽出。残りターン僅少時に GM向けOOCでソフト収束ヒント。
- テスト: 時間ブロック生成（曜日/季節境界）、イベント抽選の確率分布、SCENE_CLOSE 抽出、ソフトヒント注入。

### Phase 4: スケジューラ（自動化）
- `main.py` に `_usual_days_scheduler`（`_chronicle_scheduler` 雛形）。**1日複数スロット対応**: 冪等キーを `日付` → `日付+スロット`。
- 有効うつつセッションを各スロットで `run_scenario_turn(headless)` を `await` 回収（SSE不要）。
- 経過時間注入: 前回シーンからの経過（「前回から4時間後」）を GM へ。
- コストガード（Phase 7 のガードと統合）。
- テスト: スロット到来判定、冪等性（同スロット二重起動なし）、経過時間計算。

### Phase 5: 管理UI（設定窓口）
- `/ui/` キャラ編集に「うつつ（生活世界）」セクション: 世界設定・NPC・時間グリッド・イベントカテゴリ・GM/PCプリセット・**有効化トグル＋スロット時刻**。
- デザインは既存 `ch-*` アトムを再利用（`CLAUDE.md` CSS規約）。
- テスト: 設定の保存・読み込み、有効化トグルの反映。

### Phase 6: システムプロンプト注釈（ユーザ追加要望）
- うつつ有効キャラの **1on1 システムプロンプト**に注釈を条件付き挿入:
  「あなたはユーザに関係のない日常生活も送っており、あなたの記憶にはユーザが知らないものも含まれます」
- `request_builder` のブロックに、うつつ有効時のみ出すブロックを追加。
- テスト: 有効/無効でのブロック有無。

### Phase 7: 可視性・コストガード（秘匿＋安全弁）
- うつつセッションを通常チャット/シナリオ一覧から隠す（owner付き除外で大半カバー）。
- デバッグUI `/ui/logs` で覗ける（既存。PCターンは `scenario_chat_pc` でログ済み。必要なら feature ラベルを `usual_days` に）。
- コストガード: 1日の起動回数上限、1シーン上限ターン、`usage_recorder` 監視。claude_cli は cache 無効で毎回フル課金な点に留意。
- テスト: 一覧からの除外、ガード発火。

---

## 6. 将来拡張（今回スコープ外・着手未定）

- **うつつ中のメール送信**: うつつ進行中、キャラがユーザへメールを送る。職場から「今日ちょっと愚痴らせて」「今日雨ヤバいんだけど大丈夫?」等。常時ではなく**稀**。「仕事中に私用メールがバレると怒られる」といった制約・味付けも世界観として持たせる。送信は外向きアクションなので要・明示設計（頻度制御・宛先・キャラ判断）。
- **天気/ニュース連動**: `web_search`（Tavily、設定済み）で実世界の天気・話題を取得し、偶発イベントの種に。「今日は実際に雨」。
- **usual 記憶のラベル開示**: 想起時に usual 記憶へ「ユーザの知らない自分だけの体験」と明示ラベル。今回は content からの自然推論に任せ、キャラが共有/未共有を混同する事例が観測されたら足す（YAGNI）。

---

## 7. 未確定・実装時に判断する点

- engine_type 新設（`usual_days`）か `ensemble_pc` 流用＋無人フラグか（§4.3）
- うつつ設定の保存形式: `usual_config` JSON 単一列か個別列か
- 時間グリッド・イベントカテゴリのデフォルト値とUI入力形式（テーブル入力か自由JSONか）
- セッション永続1本か、一定期間で区切って新規起動か
- フロント `NewSessionPicker` / `api/scenario_chat/sessions` のユーザ枠必須バリデーションの有無と緩和要否

---

## 8. リスク・注意点

- **migration 前バックアップ必須**（`data/chotgor.db` / `data/lancedb/`）。
- **Windows asyncio**: うつつは LLM 呼び出しのみ。claude_cli の subprocess は既に `asyncio.to_thread` 対策済み。スケジューラは `asyncio.create_task`（既存 `_chronicle_scheduler` と同型）で問題なし。
- **コスト/レートリミット**: 無人で積み上がる。ガード（§7）必須。claude_cli は毎回フル課金。
- **backend 起動はユーザに任せる**（`feedback_no_auto_server_restart`）。動作確認は run.bat 再起動をユーザへ依頼。
- **構造を変えたら同じコミットで `docs/current-spec/ARCHITECTURE.md` も更新**（CLAUDE.md）。

---

## 9. うつつ履歴窓の再設計（v1.1・2026-07-16 追記）

### 9.1 発覚した問題

`①うつつ→②1on1→③うつつ→④1on1→⑤うつつ` の流れで、
現行の `resolve_since_dt = 最新SCENE_CLOSE時刻` だと、⑤時点で
「前回SCENE_CLOSE（＝③）以降」の1on1しか流れ込まない。②の1on1は流入しない。
④で沈黙（返事なし）だと、⑤時点でキャラから見て「返事が何もない」状態になり、
「寂しい限界OL復活」が発生する。

思想的には `external_scenes.py` 冒頭の宣言どおり **「うつつ・1on1・TRPG は同じ一本の
世界軸上で続いている時間」** であるべきで、シーン境界で見えなくなる現行は思想と乖離。

### 9.2 検討した案

**A案：未消化1on1トラッキング（棄却）**
- `chat_message` に「うつつシーンで一度でも injection されたか」フラグを持たせ、
  injection 時に立てる。未消化のものは古くても拾う。
- **棄却理由**: 「消化＝injection された時点」の運用だと、③で②が消化済みになるため、
  ⑤時点で②を再度出せず現状と変わらない。「消化＝キャラが実際に反応した時」に
  しないと機能しないが、その判定は重すぎる（キャラは黙って考えているだけかもしれない）。

**B-2案：時間窓＋メッセージ数上限（棄却）**
- `since_dt = now - N日`、`limit=500` の二段。
- **棄却理由**: 上限が付けられるのは利点だが、「キャラ設定に沿う自然なN」を
  定義しづらく、定数チューニングが要る。

**B-3案：SCENE_CLOSE時サマリ生成（保留・将来案）**
- SCENE_CLOSE時に「そのシーン＋直前1on1」を LLM で要約し蓄積。うつつには
  「過去サマリ列＋前回SCENE_CLOSE以降の生」を渡す。
- **保留理由**: 実装重（テーブル追加＋要約LLM）＋キャラの記憶パイプライン
  （WM→inscribe→narrative）と二重管理感。B-1 の上に将来足せるので今回不採用。

### 9.3 採用：B-1案（N個前のSCENE_CLOSE起点）

- `resolve_since_dt` を変更:
  - 旧: `since_dt = 最新SCENE_CLOSE時刻`
  - 新: `since_dt = N個前のSCENE_CLOSE時刻`
  - **N ＝ `usual_config.scenes_per_day`**（キャラ編集UI「1日のシーン回数（生活カレンダー用）」・既定3）
- SCENE_CLOSE が N 個未満なら **あるだけ古いSCENE_CLOSE**（＝最古の1件）まで遡る。
  それも0件なら既存フォールバック（最古うつつターン → `now - 24h`）。
- 結果：**常に直近1日ぶんの生活（うつつシーン群＋その間の1on1）が丸ごと時系列で並ぶ**。

### 9.4 採用理由

- Chotgor 思想的にもっとも軽い（キャラの記憶自律を侵さず、システム側は「窓を広げるだけ」）。
- 実装が最小（`_latest_scene_close_time` を `_nth_latest_scene_close_time(n)` へ拡張）。
- 「N シーンぶん（＝1日ぶん）重複」が思想的にも自然（人間だって前日の会話を翌日も引きずる）。
- 将来 B-3 が必要になっても B-1 と共存できる（B-1 が土台、深部はサマリで補う）。

### 9.5 トレードオフ（受容）

- 前回うつつシーン本体が毎回入るので **トークンコスト増**。うつつは環境料金
  （ユーザ課金でない）なので許容範囲。
- N＝scenes_per_day より昔の 1on1 で、シーン間に沈黙が続くと拾えない事故は残る。
  頻度は低いと想定。観測次第で B-3（サマリ）追加を検討。

### 9.6 実装箇所

- `backend/services/scenario_chat/external_scenes.py`:
  - `_latest_scene_close_time(sqlite, character_id)` → `_nth_latest_scene_close_time(sqlite, character_id, n)` へ拡張。
  - `resolve_since_dt(...)` に `scenes_per_day` 引数追加。
  - `build_all_scenes` / `build_unified_pc_messages` の呼び出し経路で `scenes_per_day` を伝搬。
- `backend/services/scenario_chat/pc_runner.py`:
  - キャラの `usual_config["scenes_per_day"]` を取り出して `build_unified_pc_messages` へ渡す。
- テスト（`tests/test_usual_days.py` など）:
  - N＝3 で 3 個前の SCENE_CLOSE 起点になる。
  - SCENE_CLOSE 数 < N で「あるだけ遡る」フォールバックが動く。
  - SCENE_CLOSE 0 件で既存フォールバックに落ちる。
  - ①③⑤（うつつ）＋②④（1on1）シナリオで、⑤時点に②④両方が並ぶ。

---

## 10. シーン起動の排他 — 1キャラにつき同時1シーン（v1.2・2026-07-27 追記）

### 10.1 発覚した問題

2026-07-26 20:30、同一うつつセッションに **2本のシーンが並行起動**し、履歴が二重トラックに
なった（GM の幕開けが 2 つ、はるの応答が 2 つ、同じ出来事に別々に反応した状態で 1 本の
`scenario_turns` に縒り合わさる）。

`run_usual_days_scene` の呼び出し元は 4 系統あり、そのうち 3 つは **それぞれ独立した毎分
`asyncio.create_task`** から駆動される:

| 呼び出し元 | 駆動元 |
|---|---|
| `main.py::_run_due_usual_scenes` | `_usual_days_tick`（毎分） |
| `main.py::_run_pending_push_resumes` | `_usual_days_tick`（毎分） |
| `services/schedule/events.py::_run_event_scene` | `_sudden_event_tick`（毎分・**別タスク**） |
| `services/actions/runner.py::_execute_scene` | `_action_tick`（毎分・**別タスク**） |

同一ティッカー内は `for` ループの逐次 `await` なので並行しないが、**ティッカーをまたぐと
排他が一切ない**。既存の `usual_scene_running_{character_id}` マーカーは 1on1 の
availability ゲート用にしか読まれておらず、シーン起動判定には使われていなかった。

### 10.2 採用：起動口での排他（先勝ち・後発は捨てる）

`run_usual_days_scene` の冒頭で `is_usual_scene_running(sqlite, owner_id)` を見て、
進行中なら**シーンを走らせずに即返す**。関数内に置くことで 4 系統すべてを 1 箇所で塞ぐ。

- check → `mark_usual_scene_running(True)` の間に `await` を挟まないため、単一イベント
  ループ上で原子的（TOCTOU にならない）。
- 戻り値に `"skipped": "already_running"` を載せ、呼び出し側は決定ログ
  （`scheduler_decisions`）に残すだけにする。
- TTL は既存の `_USUAL_RUNNING_TTL_MINUTES` に相乗り（クラッシュ時は自然失効）。

**粒度はシーン単位であってターン単位ではない。** シーン内の GM／PC ターン進行は
`run_scenario_turn(auto_advance=True)` の 1 回の呼び出しの中で完結し、
`run_usual_days_scene` を再入しないため、ガードは幕開けの 1 回しか通らない。
`scenario_chat/` 配下から `is_usual_scene_running` を参照する箇所も無い
（＝はるのターンがゲートで弾かれる経路は存在しない）。

### 10.3 検討したが採らなかった案

- **後処理まで排他を伸ばす**（マーカー解除を関数末尾へ）: あらすじ蒸留・`scene.closed`
  封筒・意図の拾い上げまでを排他対象にする案。より素直だが、1on1 の
  `unavailable("usual_scene")` が数十秒延びる。現状このシステムは 1 分未満のオーダーを
  問題にしないため**不採用**（2026-07-27 裁定）。マーカー解除は従来どおり `finally`。
- **キューイング**（後発を捨てずに待たせる）: 待っている間に前提が古くなる（経過時間メモ・
  題材 framing が陳腐化する）ため不採用。捨てて次の機会に任せるほうが自然。

### 10.4 実装箇所

- `backend/services/scenario_chat/usual_days.py`: `run_usual_days_scene` 冒頭にガード。
- `backend/main.py` / `services/schedule/events.py` / `services/actions/runner.py`:
  `skipped` の観測（決定ログ・戻り値）。
- テスト: 進行中マーカーが立っているとシーンが走らない／マーカーが無ければ走る／
  シーン内のターン進行はガードを通らない。

---

## 11. 口火の種 — 形容詞×名詞のランダム合成（v1.3・2026-08-01 追記）

### 11.1 発覚した問題

はるのうつつが **「依頼・トラブルの連鎖」に一本化**していた。3 週間分（7/9〜7/29）の
`synopsis_auto` が全て仕事案件の進捗で埋まり、本人が全件を解決してしまう
「職場のスーパーマン」化が進行。リアリティを損ない、本人も疲弊する状態になった。

原因は口火帯（`_build_usual_gm_appendix` の `is_first_gm`）の**二択構造**にある:

| 抽選 | 渡すもの | 実際に起きること |
|---|---|---|
| ヒット（`event_probability`＝はるは 10%） | `event_categories` から1つ | カテゴリ 12 件中 6 件が仕事トラブル系 → 依頼が来る |
| ハズレ（90%） | `_USUAL_LIGHT_OPENING` | 「出来事・来客・事件・NPC とのやりとりを自分から立ち上げず」 |

ハズレ側の縛りが効きすぎて、**人のいる場所から人が消える**。職場で人が消えれば
本人に残る行動は仕事しかなく、あらすじに積まれた未処理案件へ手が伸びる。
つまり口火は「依頼が来る」か「黙々と仕事の続き」の二択で、
**日常の多様な空気（雑談・疎外・ブルシットジョブ・人間関係の湿度・食・体調・気分）が
発生する余地が構造的に無かった**。

副次的に、GM が本人の対応を称揚する描写も常態化していた（7/31 の会議シーンで
Narrator が「営業部員たちは気圧されたように沈黙した」「はるの言葉が明確な道筋を
指し示した」と地の文で書いている）。プロンプトの「持ち上げない」規則は
**NPC の会話温度**の節にしかなく、地の文には効いていない。

### 11.2 採用：形容詞×名詞をコード定数から合成して渡す

シーン頭で **形容詞（重み付き）× 名詞（均等）を 1 組だけ抽選**し、GM へ渡す。

```
_SEED_ADJECTIVES = (("普通の", 60), ("いい", 8), ("悪い", 8),
                    ("楽しい", 6), ("悲しい", 6), ("うれしい", 6), ("ムカつく", 6))
_SEED_NOUNS = ("日常", "匂い", "雰囲気", "予感", "思いつき", "報せ", "人間関係")
```

役割分担 — **種を渡すのはシステム、シチュエーションを作るのは GM、
それに反応する・働きかけるのは本人**。GM は 2 語から状況を即興し、情景・人・物として
場に置く。本人がそれに乗るか、聞き流すか、気づかないかは本人の領分（`_USUAL_GM_STANDING`
の「内心・感情・選択に踏み込まない」と一貫）。

設計上の要点:

- **カタログ圧が構造的に発生しない。** 具体的な題材が設定のどこにも存在せず、
  GM が一度に見るのは 1 組だけ。「リストから選ぶ」状態にならない。
- **「普通の」60% が無風の回を確保する。** 旧 `_USUAL_LIGHT_OPENING` の役割は
  「普通の×日常」が引き継ぐ（毎回種を渡しても「毎回何か起きる」にはならない）。
- **語彙が抽象語なのでキャラ非依存。** 中世ファンタジー（柊なお）でも通用するため、
  キャラごとの設定値にする必要がない＝コード定数で持つ。
- `event_categories` / `event_probability` は**③週次突発専用**に役割を狭める。

### 11.3 検討したが採らなかった案

- **題材リストを世界設定テキスト（`scenarios.scenario`）に列挙**: 世界設定は毎レスポンス
  全文がプロンプトに入るため、書いた瞬間に「この中から選ぶリスト」になる。GM が列挙から
  選ぶだけになり、かえって多様性が縮む（**カタログ圧**）。不採用。
- **題材リストを `usual_config` の設定値＋UI で持つ**: 抽選なのでカタログ圧は出ないが、
  語彙が抽象語でキャラ固有性を持たないため、設定項目と UI を増やす利得がない。
  コード定数で足りる。不採用（2026-08-01 裁定）。
- **`event_probability` を上げて口火のヒット率を稼ぐ**: 同じ確率値が③週次突発の伏せ枠
  配置（`place_weekly_hidden_events`）でも読まれるため、突発が同時に増える。
  結局プールと確率を分ける話になる。不採用。
- **強度を下げる指示文を GM へ常設**（「難度を本人がちょうど解ける所に調整するな」
  「称揚を常態化するな」等）: スーパーマン化には直接効くが、禁止の積み上げは多様性を
  殺す懸念が強い。**保留** — 種システムで多様性を確保した後に、称揚の常態化が残って
  いれば再評価する。
- **③突発の圧ガイドの緩和**（`events.py` の「本人の予定を潰すなら 強〜激強」）:
  週 0〜3 回なので体感への寄与が小さく、見送り。ただし
  **「休日に仕事の突発が来る」問題は残課題**（7/25 土・7/26 日に連発）。
  `_usual_event_categories` は既に dict を受け付ける建付けなのに
  バケツ選択が未実装なので、ここを「平日／休日」バケツとして実装するのが筋。**宿題**。

### 11.4 実装箇所

- `backend/services/scenario_chat/usual_days.py`: 種プール定数・抽選関数・GM への提示文。
  `_USUAL_LIGHT_OPENING` / `roll_usual_event` / `_format_usual_event_hint` は役目を終えて削除。
- `backend/services/scenario_chat/service.py`: 再エクスポートの整理。
- テスト: 重み付き抽選の分布（「普通の」が多数派）、決定論乱数での再現、
  シーン頭以外では種が出ないこと、`[SCENE_CLOSE]` を口火で打たない指示が残ること。
