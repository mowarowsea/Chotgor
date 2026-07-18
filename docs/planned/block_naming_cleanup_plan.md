# システムプロンプトブロック 命名リファクタリング 計画

策定: 2026-07-18
**前提**: `switch_angle_removal_plan.md` が実施済みであること。

## 目的

システムプロンプトを構成する「ブロック」の呼称と、その内容を反映していない
残留識別子（`provider_additional_instructions` / `_build_provider_extra_block`
/ `{block_provider_extra}`）を、現在の実体に合わせて整理する。

対象は 2 系統:

1. **`provider_extra` → `session_frame` へのリネーム**
   （名は体を表していない残骸を今の意味に合わせる）
2. **「Block N」呼称の廃止**
   （名前付き placeholder が既にあるのに、docstring / コメントで番号ラベルを
   別レイヤーとして貼っており、乖離の温床になっている）

いずれもコード実体の挙動は変えない。docstring / 引数名 / 変数名 / placeholder
名の付け直しと sweep のみ。

## 経緯（実装参考用・コードコメントには残さないこと）

`provider_additional_instructions` および Block 5 (`_build_provider_extra_block`)
は、元来「キャラクターごとに、特定プリセット向けの追記指示を UI で入力できる」
機能の受け皿として作られた。キャラ設定画面の
「Additional Instructions — このモデル向けの追記指示（省略可）」欄が入口で、
`enabled_providers[preset_id].additional_instructions` に保存された文字列が
`ChatRequest.provider_additional_instructions` を経由してシステムプロンプト
Block 5 に埋め込まれる、という流れだった。

その後、シナリオ PC モードとうつつ PC が導入されたとき、キャラに配役や
「今はユーザと向き合っていない時間である」といった**セッションの枠組み**を
伝える必要が生じた。実装時に**独立ブロックを新設せず**、既存の Block 5
（`provider_additional_instructions`）に相乗りさせる判断が下された
（pc_runner.py 内コメントより: "Block を新設しない理由: 1on1 とプロンプト
構造を共有したいため。新 Block を作ると全テンプレートに分岐が漏れる"）。
`_PC_ROLE_PREAMBLE_TEMPLATE` / `_PC_USUAL_PREAMBLE_TEMPLATE` から組んだ
preamble を、`request.provider_additional_instructions` に代入する形。

そして 2026-07 の `switch_angle` 機能撤去と同時に、per-preset 追記の UI 欄も
撤去された。この時点で `provider_additional_instructions` 経路の (a) 「provider 追記」
用途は消え、(b) 「セッション枠組みメモ」用途だけが残った。名前と実体の乖離が
固定化されたため、本 PR でリネームする。

**この経緯はコード上には残さない**。実装時の理解のためにここに書いてあるだけで、
リネーム完了後は「`session_frame_instruction` はシナリオ PC / うつつ PC が
配役や日常時間の枠組みをキャラに伝えるための注入口」という一文が
docstring として残っていれば十分。歴史（`switch_angle` があった・per-preset
追記があった）は語らない。

## Part 1: `provider_extra` → `session_frame` リネーム

### 新しい命名

| 旧 | 新 |
|----|----|
| `ChatRequest.provider_additional_instructions` | `ChatRequest.session_frame_instruction` |
| `_build_provider_extra_block(...)` | `_build_session_frame_block(...)` |
| プレースホルダ `{block_provider_extra}` | `{block_session_frame}` |
| pc_runner.py 内 `merged_additional` | `session_frame` |
| pc_runner.py 内 `preamble` | 維持（局所変数のため意味は明確） |
| システムプロンプト見出し `## エンジン（モデル）固有の指示` | `## セッションの枠組み` |

### 変更ファイル

- **`backend/services/chat/models.py`**
  - `ChatRequest.provider_additional_instructions` → `session_frame_instruction`
  - フィールドコメントを「シナリオ PC / うつつ PC が、配役や『今はユーザと
    向き合っていない時間』などの**セッション枠組み**をキャラに伝えるための
    注入口。1on1 では通常は空文字」に更新
- **`backend/services/chat/request_builder.py`**
  - `_build_provider_extra_block()` → `_build_session_frame_block()`
    docstring を「シナリオ PC / うつつ PC 向けのセッション枠組みブロック」に更新
  - 見出し文字列 `## エンジン（モデル）固有の指示` → `## セッションの枠組み`
  - `DEFAULT_CHAT_SYSTEM_PROMPT_TEMPLATE` の `{block_provider_extra}` →
    `{block_session_frame}`
  - `build_system_prompt(...)` の引数名 `provider_additional_instructions` →
    `session_frame_instruction`
  - モジュール docstring のブロック説明部分を Part 2 と合わせて書き直し
- **`backend/services/chat_flow/preparation.py`**
  - `build_system_prompt(...)` 呼び出しのキーワード引数を追従
- **`backend/services/chat/request_factory.py`**
  - `build_character_request` の `ChatRequest(...)` 引数を追従
    （`switch_angle_removal_plan` 実施後は空文字リテラルが入っているだけの状態）
- **`backend/adapters/openai/router.py`**
  - 同上
- **`backend/services/scenario_chat/pc_runner.py`**
  - `merged_additional` → `session_frame`
  - `request.provider_additional_instructions = merged_additional` →
    `request.session_frame_instruction = session_frame`
  - モジュール上部のコメント (元 `line 51-54`) を「シナリオ PC 用の
    セッション枠組みをここで組み立て、`ChatRequest.session_frame_instruction`
    経由でシステムプロンプトの `{block_session_frame}` に埋め込む」に書き換え
    （「Block 5」呼称の言及は Part 2 とあわせて除去）
- **テスト追従**
  - `tests/test_chat_service.py:53` — アサーション名を追従
  - `tests/test_prompt_snapshots.py:122` — キーワード引数名を追従
  - スナップショット再生成: `tests/snapshots/prompts/system_*.txt`
    （見出し文字列が変わるため）
  - 他 grep で残った参照を追従

## Part 2: 「Block N」呼称の廃止

### 方針

- コード側の識別子（`_build_*_block()`、`{block_*}` placeholder）は
  **既に名前ベース**。触らない（意味は明確）。
- **docstring・コメント・ドキュメント上の「Block N」ラベル**だけを廃止する。
- 「配置順を語りたい場合」は名前で参照する:
  - ❌ 「Block 5 に畳み込む」
  - ✅ 「セッション枠組みブロック（`{block_session_frame}`）に畳み込む」
  - ❌ 「Block 6-8」
  - ✅ 「ワーキングメモリ関連ブロック（`{block_wm_all}` / `{block_wm_fixed}` /
    ターン注釈 WM heat 想起）」
- **番号は再導入しない**。今後の追加ブロックも名前で呼ぶ。
- **モジュール docstring の「ブロック一覧」は残す**が、番号列挙をやめて
  「安定ブロック（システムプロンプト）」と「変動ブロック（ターン注釈）」の
  2 カテゴリで名前列挙する形に書き換える。

### 変更対象ファイル

以下すべてで「Block N」「Block N-M」「Block Na」表記を、名前ベースの参照
（`{block_xxx}` またはブロック意味の平文）に置換する。番号は消す。

- **`backend/services/chat/request_builder.py:11-25`**
  - モジュール docstring のブロック一覧を書き換え:
    - 「安定ブロック（`build_system_prompt` が組み立てる部分）」→
      `{block_character}` / `{block_user}` / `{block_face_to_face}` /
      `{block_usual_days}` / `{block_session_frame}` / `{block_wm_all}` /
      `{block_wm_fixed}` / `{block_inner_narrative}` / `{block_memory_notice}` /
      `{block_chotgor_guide}` を意味と共に列挙
    - 「変動ブロック（`build_turn_annotation` が最新 user メッセージへ付加）」→
      想起記憶 / 時刻 / URL fetched / WM heat 想起 / 前回の期待
- **`backend/services/character_query.py`** `line 8-9, 78-80, 85, 135-137, 259, 262`
  - 「Block 6」→「全スレッド一覧（`{block_wm_all}`）」
  - 「Block 7」→「emotion/body/relation 固定注入（`{block_wm_fixed}`）」
  - 「Block 8」→「WM heat 想起（ターン注釈側）」
- **`backend/batch/chronicle_job.py:28-29`**
  - 同上の置換
- **`backend/batch/forget_job.py:6-7, 94`**
  - 同上の置換
- **`backend/services/scenario_chat/pc_runner.py:4, 52`**
  - `line 4`「Block 1-10」→「1on1 と同じシステムプロンプトを共有して」
  - `line 52` は Part 1 で書き換え済み
- **`backend/services/chat_flow/flow.py:106`**
  - 「Block 5」→「セッション枠組みブロック（`{block_session_frame}`）」
    ※ ただし該当行は `_build_switched_request` 内で、
    `switch_angle_removal_plan` で削除済みのはず。念のため grep で残っていないか確認
- **`backend/templates/character_edit.html:99`**
  - 「Block 1 — このキャラクターは何者か」→
    「このキャラクターは何者か（キャラクター設定の中核）」
- **`docs/current-spec/ARCHITECTURE.md:151-153`**
  - 各行の「Block 2 / 4 / 6-7 / 8」を名前ベースに書き換え
- **`docs/fable_view.md:126`**
  - 「WMブロック（Block 6-8）」→「WMブロック（全スレッド一覧・固定注入・
    heat 想起）」
- **テスト**
  - `tests/test_system_prompt.py:43, 165` — 「Block 1」「Block 9-10」等を平文化
  - `tests/test_chat_service.py:638` — 「Block 8」を「WM heat 想起」等に
  - `tests/test_forget.py:428-429, 472, 474`
  - `tests/test_ghost_model_basic.py:276-277, 309, 311`
  - 各テストの docstring / コメントを名前ベースに書き換え。**アサーション本体は
    触らない**（挙動テストは無関係）

### 明示的にやらないこと

- コード側の placeholder 名や関数名の番号化・逆番号化は一切しない
  （既に名前ベースで健全）
- 過去のコミットメッセージや PR 本文の「Block N」表現は放置
  （履歴を書き換えない）

## 実施順序

1. Part 1 のリネーム（機械的な rename → grep 追従 → テスト実行）
2. Part 2 の docstring/コメント sweep（全 15 ファイル 前後を一気に）
3. スナップショット再生成
4. `pytest` フルラン
5. 手動確認: 1on1 / シナリオ PC / うつつ PC の各経路で応答が通ること
   （プロンプト見出し `## セッションの枠組み` が期待どおり出ることも 1 件確認）

## 検証

- `pytest` 全緑
- `grep -rE "Block \d|ブロック \d" backend/ docs/ tests/` の結果が
  ゼロ件になること（許容例外: `docs/old/` は放置）
- `grep -r "provider_additional_instructions\|provider_extra\|_build_provider_extra_block" backend/ tests/`
  の結果がゼロ件になること
- 手動で 1on1 チャット・シナリオ PC ターン・うつつ PC ターンをそれぞれ
  1 ターン実行し、システムプロンプトのブロック位置が壊れていないことを
  Logs 画面で確認

## 対象外

- 他ブロックのリネーム（`{block_wm_all}` 等は既に十分な名前）
- テンプレートの構造変更（ブロックの追加・順序変更・分割は本 PR で扱わない）
- 過去ログ・コミット履歴中の「Block N」表現の書き換え
