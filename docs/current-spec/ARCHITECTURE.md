# Chotgor システム地図（ARCHITECTURE.md）

調査・実装の際に「どこに何があり、処理がどう流れるか」を素早く掴むための地図。
規範・哲学・運用ルールは `CLAUDE.md` を参照（この文書は構造の記述に徹する）。

> **鮮度について**: この文書は実装変更で腐る。ディレクトリ構成・処理フローを変える
> 変更を入れたら、この文書も同じコミットで更新すること。
> 最終更新: 2026-07-10（予報パネル・決定ログ・heartbeat 計器／ntfy 通知）

---

## 1. プロセス構成（実行時トポロジー）

```
┌──────────────────────────┐      ┌─────────────────────────────────┐
│ React frontend (vite)    │      │ Chotgor backend (FastAPI)        │
│ localhost:3000           │─────▶│ Windowsホスト直実行 port 8000    │
│ /api /v1 /ui をproxy     │      │ 起動: run.bat                    │
└──────────────────────────┘      │                                  │
                                  │  ├ 管理UI (Jinja2)  /ui/         │
┌──────────────────────────┐      │  ├ OpenAI互換API    /v1/         │
│ Claude Code CLI          │      │  ├ チャットAPI      /api/...     │
│ (claude_cliプロバイダー) │      │  └ 夜間バッチスケジューラー      │
│  └ mcp_server.py         │─────▶│     (Chronicle / Forget)         │
│    (stdio, 薄いproxy)    │ HTTP └───────┬─────────────┬───────────┘
└──────────────────────────┘              │             │
                                          ▼             ▼
┌─────────────── Docker ────────────┐  ┌────────────┐ ┌─────────────┐
│ ollama (chotgor-ollama-1) :11434  │  │ SQLite     │ │ LanceDB     │
│ infinity (embedding)      :7997   │  │ data/      │ │ data/       │
└───────────────────────────────────┘  │ chotgor.db │ │ lancedb/    │
                                       └────────────┘ └─────────────┘
```

- **backendはWindowsホスト直実行**（Docker不可。uvicornのSelectorEventLoopでは
  `asyncio.create_subprocess_exec` が使えないため、subprocessは
  `asyncio.to_thread(subprocess.run, ...)` で回避している）。
- **mcp_server.py はbackendの薄いプロキシ**。Claude CLI がスポーンし、
  `/api/mcp/tools` 経由でツールを動的取得して中継する。自前で LanceStore を開かない。
  キャラクターIDなどの状態は環境変数（`CHOTGOR_CHARACTER_ID` 等）→HTTP で伝搬する
  （Pythonのプロセス内状態はCLI越しには届かない）。
- **infinity が落ちると記憶系が縮退する**。embedding 失敗は `EmbeddingError` として
  service 層へ伝播し、ユーザには UI スケッチ欄の `recall_error`、キャラクター本人には
  システムプロンプト内の運用告知ブロック（`request_builder._build_memory_notice_block`）で
  通知される。切り分けはまず `curl http://localhost:7997/`。

## 2. リポジトリ地図（ディレクトリ責務）

### backend/

| パス | 責務 |
|---|---|
| `main.py` | エントリポイント。lifespan でストア・マネージャー・ChatService を初期化し `app.state` に集約。Chronicle/Forget/うつつ（Usual Days）スケジューラーもここで起動 |
| `api/` | HTTPエンドポイント層（ルーター）。下の「APIルーティング一覧」参照 |
| `api/ui/` | 管理UI（Jinja2 サーバーサイドレンダリング）。characters / memories / presets / scenarios / settings / instruments（計器）/ forecast（予報）/ timeline（ダイヤル） |
| `api/logs_ui/` | デバッグログ閲覧UI（`/ui/logs`）と JSON API（`/api/logs`） |
| `services/` | ビジネスロジック層。チャット・グループ・シナリオ・記憶・キャラクター問い合わせ |
| `providers/` | LLMプロバイダー抽象層。`base.py`（`BaseLLMProvider`）＋ anthropic / claude_cli / google / ollama / openai / openrouter / xai の7実装。`registry.py` が生成・ディスパッチ |
| `repositories/` | 永続化層。`sqlite/`（ORM・migration・機能別 store mixin）と `lance/`（ベクトルストア、テーブル別 ops）、`embeddings.py`（embedding プロバイダー） |
| `character_actions/` | キャラクターが使うツール（inscribe / recall / carve / switch / WMスレッド操作…）の定義・タグ抽出・実行 |
| `adapters/openai/` | OpenAI互換API（`/v1/models`, `/v1/chat/completions`）。外部クライアント向けの残存経路 |
| `batch/` | 夜間バッチ。`chronicle_job.py`（WM棚卸し・蒸留、設定時刻デフォルト03:00）と `forget_job.py`（長期記憶の忘却、04:00固定） |
| `lib/` | 横断ユーティリティ。`tag_parser`（非tool-useプロバイダーのタグ抽出・**現役**）、`debug_logger`、`time_awareness`、`web_fetch`、`log_context`、`usage_recorder`（LLM使用量記録）、`tool_event_recorder`（ツール実行イベント記録 → `tool_call_events`。Logs画面のツール使用表示の source of truth） |
| `mcp_server.py` | Claude CLI 用 MCP stdio サーバー（backendへのHTTPプロキシ） |
| `templates/` + `static/` | 管理UIのJinja2テンプレートと `chotgor.css`（デザインシステム。規約は CLAUDE.md） |

### services/ の内訳

| パス | 責務 |
|---|---|
| `services/chat/` | 1on1チャット本流。`service.py`（フロー制御）、`request_builder.py`（安定ブロック＝システムプロンプト＋変動ブロック＝ターン注釈の二層組み立て）、`request_factory.py`、`content.py`、`indexer.py`（履歴を LanceDB `chat_turns` へ upsert）、`models.py` |
| `services/group_chat/` | グループチャット。`director.py`（司会が次の発言者を決定）→ `service.py`（指名キャラを順にストリーミング生成） |
| `services/scenario_chat/` | シナリオ（TRPG風）チャット。`engine.py`（SceneEngine 抽象）、`pc_runner.py`（PCスロット駆動）、`prompt_builder.py`、`synopsis.py` / `auto_synopsis.py`（あらすじ）、`turns.py`、`mention.py` |
| `services/memory/` | 記憶管理。`manager.py`（InscribedMemoryManager: SQLite=メタデータ source of truth、LanceDB=ベクトルの協調）、`working_memory_manager.py`（WMスレッド）、`decay.py`（時間減衰の共通数式）、`reindex_service.py`（embedding変更時の全再構築） |
| `services/character_query.py` | **「キャラクターに聞く」共通入口**。バッチ処理など通常チャット以外からの問い合わせを、1on1同等のシステムプロンプト（WMブロック込み）で実行する。`ask_character` / `ask_character_with_tools`（`return_response=True` で応答テキストも取れる） |
| `services/timeline/` | **めぐり（巡り / Aliveness）の投影層＋予報層**。封筒正本（timeline_events）を観測者クラス（self / world_frame / user_ui）別の可視性ポリシーでフィルタする `projector.py`。GM への「現実の接触の記録」ブロックもここ。`forecast.py` は予報パネルの集約純関数（診断・カレンダー・圧力の無風外挿・配達シミュレータ。LLM 不使用） |
| `services/instruments/` | 計器（監査層）。Tier 1 巡回インバリアント・Tier 2 スメル検知器（正規表現）・Tier 3 判定巡回（LLM サンプリング）。アラームは `lib/instrument_recorder.py` 経由でどこからでも発火できる |
| `services/pressure/` | 圧力（社会圧・退屈圧・体調圧）。封筒の導関数として毎回計算する純関数（保存しない）＋体質インタビュー（`pressure_profile` 初期化） |
| `services/intents/` | 意図（「〜したい」の経済層）。意図圧の読み取り時計算・失効/不満化の候補挙げ・拾い上げ（Chronicle 同乗＋うつつ完走後） |
| `services/gate/` | 応答可能性ゲート。`check_availability` 純関数（従来経路: 対面 > away > うつつ進行中 > 生活時間割 ／ 生活カレンダー経路: 対面 > away > schedule_entries 占有圧最大 > OnTime）・メッセージ預かり（escrow）・能動配達（従来: 復帰＋ジッター ／ 生活カレンダー: チェック間隔格子＋決定論 reply_rate 判定 `resolve_delivery_due`）・疲労離席の発火式 |
| `services/schedule/` | **生活カレンダー（Living Schedule）**。`plan_parser.py`（[PLAN]/[EVENT] 行パーサ・24時超え表記・テンプレ裸変換・配達値個別上書き）・`weekly_batch.py`（週次バッチ①GM生成→②本人問い合わせ→schedule_entries template 層入れ替え＋③伏せ枠配置。層フォールバック=前週→テンプレ裸。冪等キー=キャラ別対象 ISO 週）・`scene_selection.py`（②導出のうつつシーン選出＝占有圧上位50%＋ランダムの決定論純関数）・`events.py`（③世界突発の確率配置＝pending 伏せ枠・発火時 GM 具体化→轢き判定（占有圧最大が勝つ）→insert→シーン）・`dilemma.py`（玉突き裁定＝③に轢かれた予定を本人が cancel/reschedule/不満化） |
| `services/actions/` | 会話外行動権。閾値評価（無料）→本人問い合わせ→実行（push / 調べもの / 臨時うつつ）→帰還のループ |

### character_actions/ の内訳

| モジュール | ツール / 役割 |
|---|---|
| `executor.py` | ツールスキーマ定義と ToolExecutor（tool-use 実行の中枢） |
| `inscriber.py` | `inscribe_memory` — 長期記憶への刻み込み |
| `recaller.py` | `power_recall` — 能動的記憶検索 |
| `carver.py` | `carve_narrative` — inner_narrative の自己書き換え |
| `threader.py` | `post_working_memory_thread` / `open_working_memory_thread` — WMスレッド操作（最高頻度） |
| `switcher.py` | `switch_angle` — プリセット（エンジン）切り替え |
| `web_searcher.py` | `web_search` — Tavily 経由の外部検索 |
| `leaver.py` | `take_leave` — 本人宣言の離席（away 設定＋chat.farewell 封筒） |
| `messenger.py` | `reach_out`（うつつ専用・現実へのプッシュ送信＋visit=対面ON＋うつつポーズ要求）／ `visit_user`（1on1専用・対面モードON）。push 実体は `services/actions/runner.execute_push` を共有。日次予算は escrow_delivery_daily_cap と共有 |
| `rescheduler.py` | `override_schedule`（1on1専用・当日予定の一時上書き）。state=OnTime/haru/adhoc/occupancy0.85 のエントリを insert するだけ（占有圧最大が勝つ読み取り解決）。`parse_until_time` は 24時超え表記対応・常に24h以内 |
| `context_tools.py` | **コンテキスト別ツール出し分けの単一判定点**。reach_out=うつつのみ（cap到達日は非露出）／visit_user=1on1かつ対面OFF／override_schedule=1on1かつ生活カレンダー有効。消費者は3系統: ①flow.py→provider.extra_tools（in-process tool-use）②mcp_server.py→GET /api/mcp/tools?character_id&origin&session_id（claude_cli）③flow.py→build_system_prompt(context_tool_hints) |
| `anticipator.py` | `[ANTICIPATE_RESPONSE:]` タグ抽出（次の展開への期待） |
| `farewell_detector.py` | 退席判定（judge LLM がキャラクターの感情状態を毎ターン外部判定。judge プリセットは `judge_preset_id`） |
| `character_context.py` | 通常チャット以外でキャラクターとして問い合わせる際の共通コンテキストブロック構築 |
| `tool_tags.py` | ツール名⇔タグ名・ログ表示ラベル/色の集約 |

### frontend/src/

| パス | 責務 |
|---|---|
| `App.tsx` | ルートコンポーネント（状態はフックへ抽出済み） |
| `api/` | backend 呼び出し層。`chat.ts` / `scenario.ts` / `logs.ts` / `translate.ts` / `sse.ts`（SSE共通処理） |
| `hooks/` | 状態管理フック。`useSessions` / `useChat` / `useGroupChat` / `useScenarioChat` ほか |
| `components/` | UI。`ChatBubbles/` と `ScenarioChatView/` は責務別ディレクトリに分割済み |

### data/（gitignore対象）

| パス | 内容 |
|---|---|
| `data/chotgor.db` | SQLite。キャラクター・セッション・記憶メタデータ・設定の source of truth |
| `data/lancedb/` | LanceDB。`inscribed_memories` / `chat_turns` / `definitions` / `working_memory_threads` の4テーブル（単一テーブル＋`character_id` フィルタ方式） |
| `data/uploads/` | チャット添付画像 |

## 3. 主要処理フロー

### 1on1チャット（本流）

```
frontend useChat
  → POST /api/chat/sessions/{id}/messages/stream   (api/chat.py)
  → ChatService (services/chat/service.py)
      1.  長期記憶を想起（RAG）→ Block 2（ターン注釈側）
      1b. ワーキングメモリ取得 → Block 6-7（システム）/ Block 8（ターン注釈側）
      2.  メッセージ内URLの自動fetch → Block 4（ターン注釈側）
      3.  request_builder が二層で組み立て（プロンプトキャッシュ対応 docs/planned/prompt_cache_plan.md）:
          - build_system_prompt: 安定ブロックのみ（前提/キャラ/ユーザ像/WM一覧・固定/inner_narrative/ガイド）
          - build_turn_annotation + append_turn_annotation: 変動ブロック（想起記憶/時刻/fetched/
            WM heat想起/圧力・意図/前回期待）を最新userメッセージ末尾へ注釈として付加
            （LLMリクエスト限り。DB履歴には残らない）
      4.  providers/registry 経由でプロバイダーへディスパッチ（SSEストリーミング）
      5.  応答からツールタグ/tool-use を処理（Inscriber / Carver / Anticipator…）
      6.  履歴を SQLite 保存 + indexer が LanceDB chat_turns へ upsert
      7.  debug_logger がログ記録
```

- tool-use 対応プロバイダーはネイティブ function calling、非対応（Claude CLI / Ollama 等）は
  `lib/tag_parser.py` による `[TAG:...]` 抽出でツールを実行する（二経路ある点に注意）。
- claude_cli プロバイダーだけは特殊で、CLI を subprocess 起動し MCP（mcp_server.py）経由で
  ツールが backend に折り返してくる。
- ツール実行は両経路とも `lib/tool_event_recorder.py` が `tool_call_events` テーブルへ
  実行時記録する（tool-use 経路は `ToolExecutor.execute()` の関門で、タグ経路は各
  `*_from_text` で記録）。Logs 画面のツール使用表示はこのイベントを読むだけで、
  生ログの逆解析（`api/logs_ui/tag_extract.py`）は 2026-06-11 以前の過去ログ互換
  フォールバックに降格済み。claude_cli の MCP 経路はプロセス越境で ContextVar が
  届かないため、`CHOTGOR_LOG_CONTEXT` env → HTTP のリレーで request_id を伝搬する。

### グループチャット

```
frontend useGroupChat → /api/group/... (api/group_chat.py)
  → director.py が司会プリセットに次の発言者を問い合わせ（失敗時はユーザーターンへ）
  → service.py が指名キャラクターを順に1on1相当でストリーミング生成
```

### シナリオチャット

```
frontend useScenarioChat → /api/scenario_chat/... (api/scenario_chat/)
  → services/scenario_chat/engine.py (SceneEngine) が
    セッション状態 + NPC + 履歴 + プレイヤー発話から UtteranceDelta / TurnRecord を yield
  → PC は pc_slots に一本化（GMプロンプトは中の人が人間かAIかを区別しない）
```

### うつつ（Usual Days — キャラの無人生活モード）

ユーザ不在のあいだ、キャラが自律的に「生活」し続ける裏の世界。シナリオチャットの
無人連鎖（headless）流用。世界（時間・場・出来事）は GM が外的フレームとして与え、
何を経験し選ぶかはキャラが決める。得た体験は `origin="usual"` の記憶として残る。

```
main.py _usual_days_scheduler（60秒ループ・冪等キー=日付+スロット）
  → services/scenario_chat/service.run_usual_days_scene(session)
    → run_scenario_turn(headless=True): GM↔キャラPC を [SCENE_CLOSE] か上限ターンまで無人連鎖
      - GMプロンプトに time_context（曜日/時間帯/季節）＋偶発イベント（混合抽選）＋
        ソフト収束ヒントを注入（prompt_builder の time_context/gm_ooc_appendix）
      - PC ターンは pc_runner（1on1同等の想起・WM・inscribe）。記憶は origin="usual"
    → シーン完走後に maybe_update_auto_synopsis(force=False) であらすじ自動蒸留
```

- 不在のユーザ（GM のユーザ言動捏造を塞ぐ）: うつつ＝ユーザがそばにいない時間。狙いは
  「GM がユーザの言動を捏造する（ユーザの知らないユーザの行動）」のを塞ぐこと。観測事故＝GM が
  `@<ユーザ>:` でユーザの SMS 全文を捏造（遠隔接触の抜け穴）＋ ANTICIPATE で次ターンへ自己成就。
  ユーザは **「ターンを取らない不在の PC」** として持つ（pc_slots に slot_id="user"、
  pc_assignments に player_type="user"）が、扱いは **主語ベースの3段ルール** に一本化する
  （`_build_absent_user_block`、作戦会議 2026-06-28 で確定）:
  - **ユーザ「について」触れるのは可／ユーザ「が」動く・話すのは不可**（目的語は可・主語は不可）。
  - Narrator（地の文）はユーザを場に持ち込まない。NPC はキャラ本人が周囲へ明かした範囲
    （`characters.user_visibility_note`）でユーザを話題化してよい（質問に限らず歓迎＝自然な呼び水）。
  - ユーザの言動の中身を持ち込めるのはキャラ本人だけ（本人は現実で交流＝自分から話題に出してよい）。
  - 連絡・訪問・通知・メッセージ・電話・LINE 等の遠隔接触も「ユーザ発の出来事」として禁止。
  - `[ANTICIPATE_RESPONSE]` でユーザの言動を予想・仕込みしない（自己成就の遮断）。
  - pc_summary（名簿）からはユーザを **外す**（routing_pcs で構築）── 名簿に並べると GM が
    「いずれ登場するキャスト」と誤読する温床。一方 `suppress_names` には残し、万一 `@<ユーザ>:`
    が出ても parser 段で破棄する最終バックストップとする。
  無人ループでは `routing_pcs = [非ユーザPC]` でルーティング候補から除外しユーザにターンを回さない。
  キャラPC側へはロスター非伝達かつユーザは履歴に出ないため「ユーザPCがいる」情報は漏れない。
  `ensure_usual_session` が user 割当を生成（既存セッションには冪等に補完）。
  設定は `character_edit.html` の `usual_user_label`（呼称）/`usual_user_position`（位置づけ）/
  `usual_user_visibility_note`（周囲への開示範囲）→ `characters` テーブルが source of truth。

- データ: `scenarios.owner_character_id`（うつつ世界の所有者）＋ `scenarios.usual_config`（JSON:
  enabled/slots/time_grid/event_categories/event_probability/max_turns_per_scene/gm・pc_preset_id）。
  履歴上限は `scenarios.history_max_turns/chars` 列（うつつフォームで設定）。
  `scenario_sessions.engine_type="usual_days"`（永続1本セッション）。
- あらすじ（履歴切り捨て時の保険）: 通常シナリオはフロントが進捗バーを見てユーザが
  プリセットを選び `/synopsis/regenerate` を叩くが、うつつは無人ゆえ介入者がいない。
  そこで **シーン完走を蒸留チェックポイント** にし、`run_usual_days_scene` が
  `maybe_update_auto_synopsis(force=False)` を呼ぶ。閾値（履歴上限 × 0.5）に達した
  ときだけ実走し、スライディングウィンドウから古いターンが押し出される前に先回りする。
  プリセットはセッションの `synopsis_preset_id`（`ensure_usual_session` で GM プリセットと
  共通に記録）。PC プリセット未指定時の配役は owner キャラの `ghost_model` を既定とする。
- 記憶 origin は3値: `real`（ユーザと共有）/ `usual`（ユーザ未共有の自分の生活体験）/
  `interlude`（シナリオPCモード幕間）。想起・蒸留・忘却では同次元（由来タグ）。
- 可視性: 汎用シナリオ／セッション一覧から除外（owner付き・engine_type で除外）。
  生ログは `/ui/logs`（feature=`usual_days` / `usual_days_pc`）でのみ覗ける。
- 管理UI: `/ui/` キャラ作成・編集フォームの Chapter 3「うつつ（生活世界）」で設定・有効化
  （新規作成・編集で同一テンプレート `character_edit.html` を共用し、本体と同じフォームで保存。
  作成は `create_character`、編集は `update_character` が `_persist_usual_world` を呼ぶ）。
- コストガード: 日次起動上限（`usual_days_daily_cap`、既定24）＋1シーン上限ターン。
- 1on1 では、うつつ有効キャラのシステムプロンプトに「ユーザの知らない日常と記憶がある」注釈を挿入。
- **reach_out（本人発プッシュ）とシーンポーズ**（2026-07-11 要件①）: うつつ PC ターンに
  `reach_out` ツールが露出（context_tools.py）。執行すると行動権 push と同一経路
  （`execute_push` = 新規セッション＋キャラ発メッセージ＋ntfy）で**現実に届き**、
  settings キー `usual_push_pause_{character_id}` にポーズ要求（sent_at/resume_at=15分後/visit）
  が立つ。loop_strategies が本人の発言終了後にこれを検知して `push_paused` → stop_condition が
  シーン停止（`run_usual_days_scene` は paused_for_push=True を返し、封筒・蒸留・意図拾い上げを
  スキップ）。`main.py _run_pending_push_resumes`（うつつスケジューラ同乗・毎分）が resume_at
  到来で GM 継続（extra_first_gm_ooc「連絡から約N分経過。返事の有無は履歴から。捏造禁止」）。
  対面ON（visit=true）中・日次上限到達時は再開見送り（キーは消す・決定ログに残す）。
  ポーズ待機中は新規シーンも起動しない。visit=true は face_to_face_mode=1 も立てる
  （「突然会いに来た」）。
- **予定コンテキスト**（要件③）: `services/schedule/awareness.build_schedule_lines` が
  「いまの予定（経過分）／本人上書き中の本来予定超過／次の固定予定（あと何分）」の淡白な行を
  組み、flow.py がターン注釈 `{block_schedule}` として 1on1・うつつ PC 両方へ注入。
  world/adhoc（③伏せ枠・突発）は**未来の予定として絶対に出さない**（ネタバレ防止。
  現在進行中のものだけ「いまの予定」として出る）。

### キャラクター問い合わせ（バッチ処理の共通経路）

キャラクターの記憶に影響する処理は必ず `services/character_query.py` を通す
（1on1と同等のシステムプロンプトで「キャラクター本人に聞く」— 根拠は CLAUDE.md）。

```
chronicle_job / forget_job など
  → ask_character() / ask_character_with_tools()
  → _collect_wm_blocks() で 1on1 と同じ WM ブロックを注入して LLM コール
```

### めぐり（巡り / Aliveness — タイムラインと動機経済）

詳細仕様は `docs/planned/aliveness_plan.md`（Phase 0〜7 実装済み・2026-07-07）。骨子:

```
正本: timeline_events（封筒 = 存在・順序・相手・時刻。中身は source_table/source_id で JOIN）
  - dual-write: chat.message / scene.turn / memory.inscribed / memory.forgotten /
    memory.carved（各 store の書き込みと同一トランザクション）
  - 直書き: chat.farewell / night.* / scene.closed / memory.recalled / intent.* / action.performed
  - 巻き戻しは削除せず retracted_at マーク（不可逆性）。バックフィルは migrations.py（marker で冪等）
投影: services/timeline/projector（observer × event_type × origin → hidden/envelope/content）
  - GM への chat.*(real) は envelope 止め・intent.* は hidden。うつつ GM に「現実の接触の記録」注入
計器: alarms / meter_snapshots ＋ services/instruments（3層）＋ /ui/instruments（静音期間表示）
決定ログ: scheduler_decisions（追記型・剪定なし）— 無人機構の評価1回ごとの結果と理由
  （fired/declined/skipped/error）。「正常な沈黙（閾値未達）」と「壊れた沈黙（機構の死）」を
  事後に区別する。記録点は行動権 runner・うつつ main.py・③突発 events.py・
  能動配達 delivery.py・週次バッチ weekly_batch.py。ループ生存は settings
  `scheduler_heartbeat_*`（`main.py _beat_scheduler` が毎周上書き）＋ Tier 1
  `scheduler_heartbeat`（鮮度1時間超でアラーム）
予報: services/timeline/forecast.build_forecast → /ui/forecast（予報パネル）。
  純関数の未来時刻評価＝無風仮定の外挿。診断（availability・圧力・意図圧・cap消費・
  「いま送ったら」配達シミュレータ）・週間カレンダー（伏せ枠全開示・②導出シーン起動時刻・
  行動権スロット予報）・圧力予報72h・揺れ監査（発火散布・リズム波形）。
  チャート系列色は `--ch-viz-1〜4`（chotgor.css・両テーマ検証済み）、描画は
  static/forecast.js（vanilla SVG）。仕様: docs/planned/forecast_panel_plan.md
圧力: services/pressure — 封筒の導関数（純関数・保存しない）。pressure_profile は体質インタビューで初期化
  - 動機ブロック（圧力の淡白な一行＋active intents＋話題権）を 1on1 プロンプトへ注入（flow.py）
意図: intents テーブル＋ services/intents — 拾い上げ2点（Chronicle 同乗・うつつ完走後）、
  失効/不満化は機械が候補を挙げ本人が裁く（soured の言葉は記憶へ刻む）
ゲート: services/gate — availability 純関数・メッセージ預かり（chat_messages.delivered_at・
  次リクエスト時に時間差注釈付き配達）・疲労離席（体調圧×engagement 発火式）・take_leave ツール
生活カレンダー: schedule_entries（実現層・重なり許容・読み取り時に占有圧最大が勝つ）＋
  services/schedule（週次バッチ①②＋③伏せ枠配置・[PLAN]/[EVENT] パーサ・②導出シーン選出・
  ③突発発火＋轢き判定・玉突き裁定）。availability は (state, 占有圧, 返信率, チェック間隔)
  へ一般化（キャラ単位トグル characters.living_schedule_enabled）。1on1 同期 SSE は OnTime
  のみ・active/busy/offline は escrow →能動配達がチェック間隔格子×決定論 reply_rate で配達
  （busy 中もチェック点で配達しうる）。うつつシーンは有効キャラで②固定予定から導出（手動
  slots は無効キャラのみ）。対面は起動ガード＋聖域化（対面中は③④保留・②シーンは捨てる）
行動: services/actions — 2時間格子＋決定論ジッターで評価（main.py の _action_scheduler）、
  push / research / 臨時うつつ（characters.action_menu トグル）、帰還で fulfilled 宣言
ダイヤル: characters.timeline_dial（0〜3）を /ui/timeline で適用・切替
```

スケジューラは `main.py` に追加系: `_instruments_scheduler`（05:00 巡回）・
`_action_scheduler`（60秒ループ・2時間格子）・`_escrow_delivery_scheduler`（毎分・能動配達）・
`_weekly_schedule_scheduler`（毎分判定・日曜夜 `weekly_schedule_time` 既定 20:00 に翌週分、
コールドスタートは当週分即時）・`_sudden_event_scheduler`（毎分・③伏せ枠の発火＝GM 具体化→
轢き判定→シーン→玉突き裁定。日次上限 `sudden_event_daily_cap` 既定3）、Chronicle / Forget /
うつつは既存。うつつ `_usual_days_scheduler` は生活カレンダー有効キャラで②導出シーンへ切替。

### 夜間バッチ

- **Chronicle**（`chronicle_time` 設定、デフォルト03:00）: WMスレッドの棚卸し・蒸留。
  価値あるものは長期記憶へ「昇格」（inscribe）。
- **Forget**（04:00固定）: 時間減衰で閾値を下回った長期記憶をキャラクター自身に問うて忘却。
  昇華は carve（inner_narrative へ）。
- 三段階蒸留: **WM →（Chronicle: 昇格）→ InscribedMemory →（Forget: 昇華）→ InnerNarrative**

## 4. APIルーティング一覧

| prefix | ファイル | 用途 |
|---|---|---|
| `/api/chat` | `api/chat.py`, `api/chat_images.py` | 1on1セッション・ストリーミング・画像 |
| `/api/group` | `api/group_chat.py` | グループチャット |
| `/api/scenario_chat` | `api/scenario_chat/` | シナリオ（scenarios / sessions / stream） |
| `/api/characters` | `api/characters.py` | キャラクターCRUD |
| `/api/inscribed_memories` | `api/inscribed_memories.py` | 記憶閲覧・Chronicle 手動実行 |
| `/api/mcp` | `api/mcp_tools.py` | MCPプロキシ用ツール定義・実行（内部API） |
| `/api/translate` | `api/translation.py` | 翻訳 |
| `/api/logs` | `api/logs_ui/` | デバッグログ JSON |
| `/ui` | `api/ui/` | 管理UI（Jinja2） |
| `/ui/logs` | `api/logs_ui/` | ログ閲覧UI |
| `/v1` | `adapters/openai/router.py` | OpenAI互換（`{char_name}@{preset_name}` をモデルIDとして解決） |

名前/UUID→キャラクター・プリセットの解決は `api/resource_resolver.py` に一元化。

## 5. 関連ドキュメント

| ドキュメント | 内容 |
|---|---|
| `CLAUDE.md` | 開発規範（哲学・命名規則・LanceDB運用・CSS規約・提案スタンス） |
| `docs/explain/README.md` | ユーザー向け紹介（※2026-06-10時点で一部記述が古い。整理予定） |
| `docs/current-spec/backend_data_design.md` | SQLite 全テーブル・全カラム定義 |
| `docs/current-spec/memory_recall_algorithm.md` | 記憶想起のハイブリッド・スコアリング（類似度×時間減衰重要度） |
| `docs/current-spec/character_resident_rules.md` | キャラクター向け仕様書（記憶・日次処理がどう扱われるか） |
| `docs/planned/usual_days_plan.md` | うつつ（Usual Days — 無人生活モード）の設計・実装計画 |
| `docs/planned/aliveness_plan.md` | めぐり（巡り / Aliveness）— タイムライン正本（封筒dual-write）・可視性・計器・動機経済の詳細仕様（Phase 0〜7 実装済み。命名ははる） |
| `docs/planned/forecast_panel_plan.md` | 予報パネル — 決定ログ・heartbeat・無風外挿・配達シミュレータの設計（2026-07-10 実装済み） |
| `docs/explain/DEAR_GHOST.md` | キャラクター向けシステムガイド（世界の歩き方） |
