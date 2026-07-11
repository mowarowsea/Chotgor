# Chotgor Backend データ設計まとめ（詳細版）

デザインチーム向けに、Backendの主要なデータ設計（テーブル定義）をまとめました。チャット履歴などセッション依存の強いデータは除外していますが、提示するテーブルについては全カラムと外部キー（FK）を網羅しています。
※ JSON型のカラムについては、単純なキーバリューの場合は「KeyValue」、そうでない場合はその構造を明記しています。

---

## 1. GlobalSetting（システム設定）
システム全体の設定値を保持するキーバリュー型ストアです。アプリ全体の設定や状態を管理します。

**【全カラム】**
* `key` (String) : 【PK】 設定のキー名
* `value` (Text) : 設定値（テキスト、またはKeyValue形式のJSON）※nullable

**【現在使用されている主なキー（動的生成含む）】**
* `user_name`: ユーザの表示名（デフォルト "ユーザ"）
* `embedding_provider`: ベクトル検索プロバイダー（google / infinity / default）
* `embedding_model`: 使用するEmbeddingモデル名
* `google_api_key`: Google系APIキー
* `infinity_base_url`: ローカルInfinityサーバーのURL
* `tavily_api_key`: Web検索用Tavily APIキー
* `claude_model`: Claudeのモデル指定
* `chronicle_time`: 定期クロニクル（記憶整理）の実行時刻（例: "03:00"）
* `chronicle_last_run_date`: クロニクル最終実行日（"YYYY-MM-DD"）
* `forget_last_run_date`: 記憶忘却処理の最終実行日（"YYYY-MM-DD"）
* `enable_time_awareness`: 時間経過認識の有効/無効フラグ
* `context_window_max_chronicled`: コンテキストに含める履歴の最大数
* `translation_preset_id`: 翻訳用に使用するLLMプリセットID
* `group_director_preset_id`: グループチャットのディレクター用LLMプリセットID
* `last_interaction_{character_id}`: キャラクターごとの最終対話日時

---

## 2. Character（キャラクター）
キャラクターの人格、プロンプト、各種設定、自己認識状態などを保持する中心となるテーブルです。

**【全カラム】**
* `id` (String) : 【PK】 UUID
* `name` (String) : キャラクター名
* `system_prompt_block1` (Text) : システムプロンプト（基本設定）
* `inner_narrative` (Text) : 内的叙述（キャラクター自身の自己物語テキスト）。三段階の蒸留パイプラインの最終段であり、Forgetバッチによる昇華・凝縮の書き込み先
* `cleanup_config` (JSON) : 履歴クリーンアップ用の設定。単純な **KeyValue** 形式（例: `{"days": 30}`）
* `enabled_providers` (JSON) : 有効化されているLLMプロバイダー設定。以下の**構造を持つ**。
  ```json
  {
    "preset_id": {
      "additional_instructions": "追加のシステムプロンプト指示",
      "when_to_switch": "このモデルに切り替えるべき条件・タイミング"
    }
  }
  ```
* `ghost_model` (String) : バックグラウンドでの記憶整理（Chronicle/forget）などで使用されるLLMモデル（プリセットID）※nullable
* `image_data` (Text) : Base64エンコードされたアバター画像 ※nullable
* `switch_angle_enabled` (Integer) : 視点切り替え機能の有効化フラグ（1: ON, 0: OFF）
* `self_reflection_mode` (String) : 自己参照ループ設定（disabled / local_trigger / always）
* `self_reflection_preset_id` (String) : 契機判断モデルプリセットID（local_trigger時）※nullable
* `self_reflection_n_turns` (Integer) : 自己参照に使う直近Nターン数
* `self_history` (Text) : これまでの経緯と現在の状態（Chronicle処理で自己更新）
* `relationship_state` (Text) : ユーザ・他キャラとの関係性（Chronicle処理で更新）
* `farewell_config` (JSON) : 感情閾値や退席設定などを定義するJSON ※nullable。以下の**構造を持つ**。
  ```json
  {
    "thresholds": {"anger": 0.8, "sadness": 0.5},
    "farewell_message": {"negative": "もう限界です..."},
    "estrangement": {"lookback_days": 30, "negative_exit_threshold": 5}
  }
  ```
* `relationship_status` (String) : "active" または "estranged"（疎遠になると要求を拒否）
* `definition_embedding_id` (String) : LanceDB定義テーブル内のドキュメントID ※nullable
* `allowed_tools` (JSON) : そのキャラクターに許可された外部ツール設定。以下の**構造を持つ**。
  ```json
  {
    "web_search": true,
    "google_calendar": false,
    "gmail": false,
    "google_drive": false
  }
  ```
* `created_at` (DateTime) : 作成日時
* `updated_at` (DateTime) : 更新日時

---

## 3. InscribedMemory（保存記憶レコード）
キャラクターが自ら「保存する」と判断して残した長期記憶レコードです。三段階の蒸留パイプライン（WorkingMemory → InscribedMemory → InnerNarrative）の第2段に位置し、Chronicleバッチによる昇格先・Forgetバッチによる昇華・削除の対象となります。

**【全カラム】**
* `id` (String) : 【PK】 UUID
* `character_id` (String) : 【FK -> characters.id】 所有キャラクターID
* `content` (Text) : 記憶のテキスト内容
* `memory_category` (String) : カテゴリ（general など）
* `contextual_importance` (Float) : コンテキスト上の重要度スコア（0.0 - 1.0）
* `semantic_importance` (Float) : 意味的知識としての重要度スコア
* `identity_importance` (Float) : アイデンティティとしての重要度スコア
* `user_importance` (Float) : ユーザ関係に関する重要度スコア
* `source_preset_id` (String) : 記憶を作成したプリセットID ※nullable
* `last_accessed_at` (DateTime) : 最終アクセス日時 ※nullable
* `access_count` (Integer) : アクセス回数
* `created_at` (DateTime) : 作成日時
* `updated_at` (DateTime) : 更新日時（内容や重要度変更時のみ） ※nullable
* `deleted_at` (DateTime) : 論理削除用日時 ※nullable

---

## 4. WorkingMemoryThread（ワーキングメモリスレッド）
キャラクターの並行する短期記憶・中期記憶のストリームです。BBSのスレッドのような形式で管理されます。

**【全カラム】**
* `id` (String) : 【PK】 UUID
* `character_id` (String) : 【FK -> characters.id】 所有キャラクターID
* `type` (String) : スレッドの種類（emotion / body / task / topic / relation）
* `summary` (Text) : タイトル相当。Embeddingの検索対象になる
* `atmosphere_tag` (Text) : 質感の短いタグ（温度感や終わり方など）
* `importance` (Float) : 重要度スコア（0.0 - 1.0）
* `is_open` (Integer) : 1=Open（運用中）, 0=Archived（過去のもの）
* `relation_target` (String) : relation型のみ使用。関係相手の識別子 ※nullable
* `created_at` (DateTime) : 作成日時
* `updated_at` (DateTime) : 更新日時
* `last_touched_at` (DateTime) : 時間減衰の起点となる明示的な更新日時 ※nullable

---

## 5. WorkingMemoryPost（ワーキングメモリポスト）
スレッド内に時系列で追加される個々の書き込み（短期記憶の断片）です。

**【全カラム】**
* `id` (String) : 【PK】 UUID
* `thread_id` (String) : 【FK -> working_memory_threads.id】 親スレッドID
* `content` (Text) : ポストのテキスト内容
* `created_at` (DateTime) : 作成日時

---

## 6. SessionDrift（行動指針 / SELF_DRIFT）
チャット中にキャラクターが自らに課した一時的な行動指針です。

**【全カラム】**
* `id` (String) : 【PK】 UUID
* `session_id` (String) : 【FK -> chat_sessions.id】 所属するチャットセッションID
* `character_id` (String) : 対象キャラクターID（※参照用・明示的FK制約はなし）
* `content` (Text) : 行動指針テキスト
* `enabled` (Integer) : 有効化フラグ（1: ON, 0: OFF）
* `created_at` (DateTime) : 作成日時

---

## 7. LLMModelPreset（LLMモデルプリセット）
AIプロバイダーや利用するモデルの設定を保持します。

**【全カラム】**
* `id` (String) : 【PK】 UUID
* `name` (String) : プリセット名（例: "Google-Gemini3Flash"）
* `provider` (String) : プロバイダー名（google, openai など）
* `model_id` (String) : 利用モデル名（gemini-2.0-flash など）
* `thinking_level` (String) : 推論レベル（default / low / medium / high）
* `created_at` (DateTime) : 作成日時

---

## 8. Scenario（シナリオテンプレート）
シナリオチャットで再利用可能な世界観や導入設定のテンプレートです。

**【全カラム】**
* `id` (String) : 【PK】 UUID
* `title` (String) : シナリオタイトル
* `scenario` (Text) : シナリオ概要や世界観の自由記述テキスト ※nullable
* `intro` (Text) : 導入部分（@キャラ:記法）のテキスト ※nullable
* `user_alias` (String) : プレイヤー（ユーザ）の呼び名
* `gm_preset_id` (String) : ゲームマスターとして動くLLMプリセットID（llm_model_presets.id を参照）
* `history_max_turns` (Integer) : 送信履歴の最大ターン数（NULL=デフォルト） ※nullable
* `history_max_chars` (Integer) : 送信履歴の最大文字数（NULL=デフォルト） ※nullable
* `created_at` (DateTime) : 作成日時
* `updated_at` (DateTime) : 更新日時

---

## 9. ScenarioNpc（シナリオNPC）
シナリオ内に登場する軽量なキャラクター設定です。

**【全カラム】**
* `id` (String) : 【PK】 UUID
* `scenario_id` (String) : 【FK -> scenarios.id】 所属シナリオID
* `name` (String) : NPCの呼称（@タグで指定。シナリオ内でユニーク）
* `description` (Text) : 人物像・口調・話し方などをまとめた自由テキスト ※nullable
* `image_data` (Text) : アバター画像（base64 data URI） ※nullable
* `promoted_character_id` (String) : 既存キャラへの昇格予約ID ※nullable
* `created_at` (DateTime) : 作成日時
