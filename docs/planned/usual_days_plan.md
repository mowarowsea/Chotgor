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
| シーンの種 | ①外的フレーム（時間グリッド）＋②継続（あらすじ＋履歴＝既存）＋③偶発イベント（混合方式：頻度は機械抽選で確実制御、中身はカテゴリだけ渡してGM即興） |
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
    `event_categories`（時間帯/曜日/季節別の偶発イベント候補）, `event_probability`（偶発の発生率）,
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
