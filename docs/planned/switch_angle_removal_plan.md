# switch_angle 機能撤去 計画

> ステータス: **撤去完了（2026-07-19, `a348662`）**。策定 2026-07-18。
> 残置物: `migrations.py` の `characters.switch_angle_enabled` 列削除マイグレーションと、
> 過去ログ表示用の `tool_tags.py` メタのみ。

## 目的と背景

`switch_angle` は、キャラクター応答の途中で別プリセット（モデル）へ切り替え、
切り替え先のモデルが同ターンに再応答する機能。
当初「はるがフィルタの緩いモデルへ静かに逃げる」用途を想定して作られたが、
Claude 系モデルがほぼ NG なしで実用充分になり、想定した使い方は不要になった。
実運用でも LLM 側から呼び出されておらず、機能単位で不要と判断。

撤去することで:

- `flow.py` の**再帰ディスパッチ**構造がまるごと消え、線形処理へ縮む
- `ChatRequest.available_presets` および `_build_switched_request` の
  `replace(...)` 経路が消え、ChatRequest の役割が単純化
- システムプロンプトの Chotgor 操作ガイドが「presets 有無」で分岐しなくなる
- Claude CLI プロバイダーの MCP 経由結果を in-process へ**転写する**特別
  処理が消える（MCP 越境の複雑さの一部が畳める）

## 削除方針（決定事項）

1. **`characters.switch_angle_enabled` カラムは DROP COLUMN する**
   （マイグレーション実施・事前バックアップ必須）
2. **既存の tool_call_events / チャットログに残る `switch_angle` / `SWITCH_ANGLE` は放置**
   （ログ表示は「不明ツール」フォールバックに任せる。剪定コードは足さない）
3. **「ついで」の抽象化整理を撤去 PR に同梱する**
   （再帰ディスパッチが消えることで自然に縮む箇所のみ。追加リファクタは別提案）
4. **キャラ設定画面の per-preset 入力欄をあわせて撤去する**
   - 「Additional Instructions このモデル向けの追記指示」欄も削除
   - 「このモデルに切り替えるタイミング」欄は既に本計画に含まれる
   - `enabled_providers` JSON の per-preset 値は空 dict `{}` になる（キー存在=有効
     判定は維持）。既存レコードの余剰キーは放置（決定 2 と同方針）。
   - **`ChatRequest.provider_additional_instructions` フィールドと `Block 5`
     `_build_provider_extra_block` は残す**。理由: 名前は「provider 追記」だが
     実体は 2 系統で使い回されており、
     - (a) キャラ設定 UI の per-preset 追記 ← 今回削除
     - (b) シナリオ PC モード / うつつ PC の preamble 注入
       （`backend/services/scenario_chat/pc_runner.py:349-378`）
     の (b) が残るため。(b) は `_PC_ROLE_PREAMBLE_TEMPLATE` /
     `_PC_USUAL_PREAMBLE_TEMPLATE` から「配役の自覚」「これは本人の日常」を
     組み立て、`request.provider_additional_instructions = merged_additional`
     として代入している。Block 5 を消すとシナリオ PC / うつつが枠組み情報を
     受け取れなくなる。pc_runner.py:52-54 のコメントに「Block を新設しない理由:
     1on1 とプロンプト構造を共有したいため」と明記されているとおり、意図的な借用。
     - フィールド名が (a) の意味しか表さず誤解を招く点は事実。中立名
       （例: `extra_system_instructions`）へのリネームは意味が正しくなるが、
       本 PR の範疇（switch_angle 撤去）を超えるため**別 PR** とする。

## 撤去範囲（ファイル別 punch list）

### A. コア（tool_executor / switcher）

- **`backend/character_actions/switcher.py`** — ファイル削除
- **`backend/character_actions/executor.py`**
  - `line 25`: `Switcher, extract_switch_angle_tags` import 削除
  - `line 66`: `ToolCall.name` docstring から `switch_angle` を除去
  - `line 100`, `line 120`, `line 192`: クラス docstring / `_switcher` 記述削除
  - `line 164`: `self._switcher = Switcher()` 削除
  - `line 169-172`: `switch_request` property 削除
  - `line 194-197`, `line 199`: `record` / `source` docstring から
    `switcher` / `switch_angle` 記述を削除
  - `line 262-276` `apply_all_tags`: `apply_switch_angle_tags(clean)` 呼び出しを外し、
    `apply_carve_narrative_tags` の戻り値を直接返す形へ簡約
  - `line 240-243`: モジュールコメントから `SWITCH_ANGLE` を除去
  - `line 324-343`: `apply_switch_angle_tags` メソッド削除
  - `line 384-388`: `_dispatch` の `switch_angle` 分岐削除
- **`backend/character_actions/tool_specs.py`**
  - `line 47-50`: `SWITCH_ANGLE_SCHEMA, SWITCH_ANGLE_TOOL_DESCRIPTION` import 削除
  - `line 124`: `ToolSpec("switch_angle", ...)` 削除
- **`backend/character_actions/tool_tags.py`**
  - `line 24`: `TOOL_TO_TAG` から `"switch_angle": "SWITCH_ANGLE"` 削除
  - `line 37`: `TAG_META["SWITCH_ANGLE"]` 削除
  - `line 107-` の `SWITCH_ANGLE` フィールド分岐削除
- **`backend/lib/tag_parser.py`**
  - `line 287-288`: KNOWN_PREFIXES 相当の `[SWITCH_ANGLE:` エントリと関連コメント削除

### B. フロー / リクエスト構築（もっとも「入り組んで」いる部分）

- **`backend/services/chat_flow/flow.py`**
  - モジュール docstring `line 13`: `switch_angle / power_recall の再帰` から
    `switch_angle /` を削除
  - `_extract_switch_info` (`line 78-94`) 削除
  - `_build_switched_request` (`line 96-144`) 削除
  - `execute()` `line 197-203`: switch 分岐削除
  - `execute_stream()` `line 367-384`: switch 分岐削除（`angle_switched` yield も同時消滅）
  - `line 388` 付近のコメントから `switch_angle` 記述を除去
- **`backend/services/chat_flow/strategies/one_on_one.py`**
  - `line 65,87`: docstring/コメントから `angle_switched` / `switch_angle` 記述削除
- **`backend/services/chat_flow/preparation.py`**
  - `line 266`: `available_presets=` 引数削除（`build_system_prompt` シグネチャ変更に追従）
- **`backend/services/chat/models.py`**
  - `line 30-34`: `ChatRequest.available_presets` フィールド削除
- **`backend/services/chat/request_factory.py`**
  - `line 14-48`: `build_available_presets` 関数削除
  - `line 5, 89`: docstring から `available_presets` 記述削除
- **`backend/services/chat/request_builder.py`**
  - `line 559-594`: `_build_switch_angle_block` 削除
  - `line 599, 610, 616, 637-638, 651-652`: `_build_chotgor_block` から
    `available_presets` 引数・分岐削除
  - `line 679, 736`: `build_system_prompt` から `available_presets` /
    `current_preset_name` 引数削除（後者は switch 案内以外で使っていないため
    合わせて削除。使用箇所を grep で確認）
- **`backend/services/scenario_chat/pc_runner.py`**
  - `line 39`: `build_available_presets` import 削除
  - `line 224, 230`: docstring から `angle_switched` 削除
  - `line 340, 372`: `available_presets` 構築・引き渡し削除
  - `line 409-` の `angle_switched` 中継削除
- **`backend/services/gate/delivery.py`**
  - `line 357` の `angle_switched` 分岐削除

### C. API / SSE

- **`backend/api/chat.py`**
  - `line 27`: `build_available_presets` import 削除
  - `line 141, 151`: `available_presets` 構築・引き渡し削除
  - `line 498-504`: `angle_switched` chunk 処理削除
- **`backend/adapters/openai/router.py`**
  - `line 15, 161, 182`: 同上
- **`backend/api/utils.py`**
  - `line 9, 20`: `build_available_presets` 再エクスポート削除

### D. プロバイダー（Claude CLI）

- **`backend/providers/claude_cli_provider.py`**
  - `line 288-320`: `_extract_switch_angle_from_stream_json` 削除
  - `line 467`: docstring から `switch_angle のみ例外で〜` 記述削除
  - `line 493-503`: MCP 経由 switch_angle の in-process 転写ブロック削除

### E. DB / キャラ設定

- **`backend/repositories/sqlite/models.py`**
  - `line 114`: `switch_angle_enabled = Column(...)` 削除
- **`backend/repositories/sqlite/stores/character_store.py`**
  - `line 20, 57`: `create_character` の引数削除（`update_character` にも同型があれば同様）
- **マイグレーション**
  - 新規スクリプト `backend/repositories/sqlite/migrations/NNNN_drop_switch_angle_enabled.py`
    （通し番号は `migrations/` の最新に合わせる）
  - **SQLite の `ALTER TABLE ... DROP COLUMN` は SQLite 3.35+ で使えるが、
    互換性重視なら**「新テーブル作成 → データ移行 → 旧テーブル削除 → リネーム」の
    従来手順を採る。既存マイグレーションの流儀を踏襲すること。
  - 事前バックアップ: `data/chotgor.db` / `data/lancedb/` を退避（memory ルール）

### F. UI

- **`backend/templates/character_edit.html`**
  - `line 273-278` 付近: `Additional Instructions` フィールド（`ai_{preset.id}`
    textarea）を含む `<div class="ch-field">` 削除
  - `line 275`: `<span class="ch-hint">このモデル向けの追記指示（省略可）</span>` 削除
  - `line 279-284` 付近: `when-to-switch-field` の `<div>` 群削除
    （`wts_{preset.id}` textarea を含む）
  - `line 313-325` 付近: `switch-angle-group` ブロック全体削除
  - `line 787-795, 855-880` 付近: `updateSwitchAngleVisibility` /
    `updateWhenToSwitchVisibility` などの JS ハンドラ削除（依存する変数も追跡削除）
  - 「新プリセット追加テンプレ」側 (`line 850-861` 付近) の `Additional Instructions`
    `<div>`（`ai_${presetId}`）と `wts_${presetId}` textarea も削除
- **`backend/api/ui/characters.py`**
  - `line 44-57` 付近: `_collect_enabled_providers` を「preset_id をキーに
    空 dict を詰めるだけ」に簡約
    （`additional_instructions` / `when_to_switch` 両キーを外す。docstring も更新）
  - `line 123-136`: `create_character` フォーム受け取りから `switch_angle_enabled` 削除
  - `line 191-201`: `update_character` 側も同様
- **`backend/services/chat/request_factory.py`**
  - `line 101`: `model_config = (char.enabled_providers or {}).get(preset.id, {})`
    は残す（enabled 判定のロジックが後段にあれば維持）が、
  - `line 125`: `provider_additional_instructions=model_config.get("additional_instructions", "")`
    は常に空文字になるため **`provider_additional_instructions=""`** に置換
    （`build_character_request` の caller 側で PC モード等が上書き代入する経路は
    温存されるため、初期値の由来だけを変える）
- **`backend/adapters/openai/router.py`**
  - `line 173`: 同上、`provider_additional_instructions=""` へ置換
- **`backend/api/logs_ui/tag_extract.py`**
  - `line 115-` の `SWITCH_ANGLE` 分岐削除・`line 129` のコメント修正
- **`backend/static/chotgor.css`**
  - `tag-switch` クラスがあれば削除（grep 現時点で無し。念のため撤去 PR 時に再確認）
- **`frontend/src/api/chat.ts`**
  - `line 151-158` 付近: `{ type: "clear" }` および `{ type: "angle_switched"; ... }`
    のイベント型定義削除（`"clear"` は backend で yield されていない死コードのため
    合わせて撤去）
- **`frontend/src/api/scenario.ts`**
  - `line 187-192` 付近: `angle_switched` イベント型削除
- **`frontend/src/hooks/useChat.ts`**
  - `line 34-36`: `setSelectedModel` の `switch_angle` 用途コメント修正または引数削除
    （他用途があるなら保持、確認）
  - `line 150-160` 付近: `"clear"` / `"angle_switched"` ハンドラ削除
- **`frontend/src/hooks/useScenarioChat.ts`**
  - `line 447-450` 付近: `"angle_switched"` ハンドラ削除

### G. MCP

- **`backend/mcp_server.py`**: 現状 `switch_angle` を露出していないことを grep で確認済み
  （もしプロキシ経路 `api/mcp_tools.py` で `switch_angle` が呼び出せる状態なら、
  A の ToolSpec 撤去で自動的に消える。実装当時のメモに従い再確認する）

### H. テスト

- **削除**:
  - `tests/test_switcher.py`
  - `tests/test_switch_angle_service.py`
  - `tests/test_switch_angle_executor.py`
  - `tests/_switch_angle_helpers.py`
- **部分修正**:
  - `tests/test_tools.py`, `tests/test_tool_tags.py`,
    `tests/test_tool_event_recorder.py` — `switch_angle` ケース削除
  - `tests/test_openai_adapter.py` — `switch_angle_enabled` を扱う
    `_make_character` / `test_available_presets_*` ケース削除。
    残す `enabled_providers` の値は空 dict `{}` へ差し替え。
  - `tests/test_chat_api.py:275,405` — `enabled_providers={"preset-1": {"additional_instructions": ""}}`
    等を `{"preset-1": {}}` へ差し替え
  - `tests/test_chat_service.py:53` — `req.provider_additional_instructions == ""`
    は空文字前提が正のため維持（PC モードでない経路の期待は変わらない）
  - `tests/test_resource_resolver.py:98-105` — テストが per-preset の
    `additional_instructions` に依存する場合はテスト意図を再定義（削除機能の
    残骸なので、削るか無関係なケースに書き換える）
  - `tests/test_prompt_snapshots.py:122` — `provider_additional_instructions=`
    は Block 5 の表示テストのため引数自体は残す（値を変える必要なし）
  - `tests/test_mcp_tools_api.py` — 期待ツール一覧から `switch_angle` 削除
  - `tests/test_logs_ui_tag_extract.py` — `test_switch_angle_*` 2 テスト削除
  - `tests/test_logs_ui_tool_calls.py` — `switch_angle` function 呼び出しテスト削除
  - `tests/test_claude_cli_format.py` — `TestExtractSwitchAngleFromStreamJson`
    クラス全削除、`_extract_switch_angle_from_stream_json` import も削除
  - `tests/test_tag_parser.py` — `SWITCH_ANGLE` の言及箇所修正
  - `tests/test_format_speech.py` — `angle` 言及があるか確認して修正
- **スナップショット再生成**:
  - `tests/snapshots/prompts/system_tags_full.txt`
  - `tests/snapshots/prompts/system_tools_full.txt`
  - `tests/snapshots/prompts/system_tags_minimal.txt`
  - `tests/test_prompt_snapshots.py` の `_AVAILABLE_PRESETS` を撤去（`line 125` 付近）

### I. 計器 Tier2（判断保留 → 「残す」推奨）

- **`backend/services/instruments/tier2.py`** `line 24`:
  `_TAG_NAMES` に `"SWITCH_ANGLE"` が入っている。
  これは応答本文にツールタグ様の文字列が漏れていないか検知する用途のため、
  **残しておく**（LLM が幻覚で `[SWITCH_ANGLE:...]` を出したときの smell 検出）。
  ただし新規機能として扱わないため、コメントに「歴史的タグ・現在は未定義」の
  一言を残す。

### J. 設計書

- **`docs/current-spec/ARCHITECTURE.md`**
  - `line 93` の `services/chat_flow/` 説明から `switch_angle /` 削除
  - `line 115` の `switcher.py` 行削除
- **`docs/current-spec/character_resident_rules.md`**
  - `line 68-70` 切り替え候補記述削除
  - `line 382-` `### 7.4 switch_angle` セクション削除（節番号の詰め直しを行う）
- **`docs/explain/README.md`**
  - `line 83-84` ツール一覧の `switch_angle` 行削除
  - `line 160-162` Enabled Providers 説明の `switch_angle で切り替え可能` 表現削除
- **`docs/explain/DEAR_GHOST.md`**
  - `line 107-` `### 2.5 switch_angle` セクション削除（節番号詰め直し）
- **`docs/old/backend_data_design.md`**
  - `line 54-56` の `enabled_providers` JSON 例から
    `additional_instructions` / `when_to_switch` キー削除
  - `line 61-62` `switch_angle_enabled` 記述削除（old ディレクトリのため任意だが揃える）
- **`CLAUDE.md`**: 影響なし（switch_angle 直接記述なし）
- **`MEMORY.md`**: 「残課題」節から `switch_angle MCP対応` の記述を削除
  （関連ファイル: `project_mcp_integration.md` を開いて中の該当行も削除）

## 「ついで」の抽象化整理（Fable 想定・撤去 PR 同梱）

再帰ディスパッチが消えることで、以下は自然に片付く。**追加のリファクタは
入れず、削除の副産物として起こる整理のみ**を対象とする。

1. **`ChatFlow.execute()` / `execute_stream()` の線形化**
   - `switched = self._build_switched_request(...)` → `return await self.execute(switched)`
     の再帰枝が消えることで、両メソッドは頭から尻まで直線処理になる。
   - `text_already_streamed` フラグは、switch 用に「未 stream の第1応答テキストを
     救い上げる」ためだけに残っていたわけではない（タグ方式でも使う）ため**保持**。
     ただし switch 分岐削除でロジックは単純化する。
2. **`ChatRequest` のフィールド整理**
   - `available_presets` 削除（B 節に既出）
   - `current_preset_name` / `current_preset_id` は他用途で使われているため保持
     （logging / preset_id 記録に必要）
3. **`_build_chotgor_block` の引数整理**
   - `available_presets` 削除に伴い、`current_preset_name` が
     Chotgor ブロック生成で不要になるなら合わせて外す
     （grep で他用途を確認して判断）
4. **`enabled_providers` JSON スキーマ**
   - `{additional_instructions, when_to_switch}` の両キーを落とし、per-preset の
     値は空 dict `{}`（存在＝有効判定のみ）に集約。
   - **既存レコードに残る `additional_instructions` / `when_to_switch` キーは
     剪定しない**（放置方針）。読み出し側で無視すればよい。
   - `character.enabled_providers or {}).get(preset.id, {})` の
     `.get(key, "")` パターンは空文字リテラルに直接置換できる。
   - シナリオ PC モード (`services/scenario_chat/pc_runner.py:365`) の
     `existing_additional = (model_cfg.get("additional_instructions", "") or "").strip()`
     は常に空文字になるため、`merged_additional = preamble.strip()` に単純化
     （`existing_additional` 変数と `\n\n` 結合ロジックを削除できる）。
5. **Claude CLI プロバイダーの MCP 転写コメント**
   - `switch_angle のみ例外〜` の説明が消えることで docstring が縮む。
   - 「MCP 経由で実行されたツールを in-process 側 tool_executor へ転写する」
     経路自体はこの機能でしか使っていない可能性が高いため、削除で
     `_extract_switch_angle_from_stream_json` と一緒に消える。他ツールで
     MCP 転写が要る箇所が無いか grep で最終確認。

## 実施順序（推奨・上流 → 下流）

1. **設計書を先に更新**（CLAUDE.md 原則: 先に仕様を落とす）
   - J 節をすべて先に済ませる（コード変更前に PR プレビューでレビュー可能に）
2. **UI 表面を落とす**（新規で有効化できない状態に）
   - F 節: `character_edit.html` の入力欄・JS ハンドラ削除
   - `api/ui/characters.py` のフォーム受け取り削除
3. **フロー中枢**
   - B 節: `flow.py` の再帰削除・`request_builder` のブロック削除・
     `preparation.py` の引数削除・`ChatRequest.available_presets` 削除
4. **executor / switcher / provider 本体**
   - A 節・D 節を一気に
5. **API / SSE / フロント接続**
   - C 節・F 節（フロントエンド） を同時
6. **DB カラム DROP マイグレーション**
   - E 節。ここで **backend 再起動が必要な旨をユーザーへ伝える**
     （Claude Code から `rerun.bat` は叩かない、feedback_backend_auto_restart）
7. **テスト＆スナップショット調整**
   - H 節を最後にまとめて。スナップショットは実装完了後に再生成。

## 検証

- **単体テスト**: `pytest` 全体を実行（Windows/PowerShell 経由）。
  H 節の削除・修正で赤が全部消えていれば OK。
- **手動確認（backend 再起動後）**:
  - 1on1 チャットが通常応答すること（`switch_angle` 案内が
    Chotgor 操作ガイドから消えていること）
  - `character_edit.html` の switch 設定 UI が完全に消えていること
  - Logs 画面で過去の `SWITCH_ANGLE` タグを含むエントリを開いても
    エラーにならないこと（放置ログの回帰確認）
  - Claude CLI プロバイダー経路のキャラで、通常のツール
    （inscribe_memory / carve_narrative / power_recall）が動くこと
- **MCP 経路の smoke test**: `mcp__chotgor__*` の tools/list に
  `switch_angle` が現れていないこと

## 対象外（明示的に「やらない」）

- 過去の tool_call_events / チャットログの `switch_angle` レコード剪定
- `enabled_providers` JSON 内の既存 `when_to_switch` キー剪定
- 撤去 PR 内での追加リファクタ（Chotgor ブロック生成の全面書き換え等）
- `snapshot`（Git 管理）機能への影響検討（現状無関係）
