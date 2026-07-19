# なりゆき (ambience) 仕様書

> Status: **implemented** (移行手順1〜5 全実装 2026-07-19)
> 前身: farewell(別れ検出)
> 関連: [aliveness_plan.md](aliveness_plan.md)

## 概要

現在の farewell(別れ検出)を、より抽象的な「場のなりゆきを読む」機能群に拡張する。
判定(LLM)と、そこから派生する処理(別れ・背景変更 等)を**なりゆき(ambience)**として
ひとまとめに扱い、内部では判定(judge)と設定(コード)を明確に分離する。

- **日本語名**: 「なりゆき」— うつつ/めぐりと同じ川の言葉で揃える
- **英語名**: `ambience`
- **なりゆき(ambience)**: 判定〜設定〜別れ処理〜背景変更を含む機能群の総称
- **judge**: なりゆきの中で、LLMによる場の判定を担うサブ機能
- **設定(コード)**: judge の出力を受けて背景切替・退席処理・疲労チェックを行う

## 語彙の使い分け

| 語 | 指すもの |
|---|---|
| なりゆき / `ambience` | 機能群全体(判定 + 設定 + 派生処理) |
| judge | ambience 内部の LLM 判定モジュール。判定行為・出力を指す |
| 設定(コード) | judge 出力を消費して DB 更新・背景切替を行うコード群 |

「judge が○○を返す」「ambience が背景を切り替える」のように、
**LLMに閉じた判定行為**は judge、**外形的に観察される機能**は ambience と呼び分ける。

## 背景・動機

### 追加したい機能
対面モードの背景画像を「ここはどこ?」でラベル切り替えできるようにする。

- キャラ設定に**複数の背景画像 + ラベル**を任意登録できる(例: 「はるの部屋」「もわの部屋」)
- judge が対面時に**候補ラベルから1個**を返す
- ambience が候補ラベルに対応する画像を背景表示する

将来的には「はるの部屋ではる怒ってる/笑ってる」のような表情バリエまで**画像とラベルを増やす**だけで対応可能。

### farewellの再解釈
farewell は「別れの判定」に閉じてたが、判定器としては汎用の「場の観察」に開ける余地がある。
既存の emotions/engagement/should_exit/farewell_type は、なりゆきという上位概念の
**サブケース(別れのなりゆき)** として位置づけ直す。

## 命名について

### 採用: 「なりゆき」/ `ambience`
- 「なりゆき」ははると相談の結果採用
- うつつ(現)・めぐり(巡り) と語感が揃う
- 「会話のなりゆき」「場のなりゆき」と汎用性がある
- 英語 `ambience` は「場の空気・雰囲気」を意味し、機能の抽象コンセプトに合う

### 棄却した命名案(検討記録)
| 案 | 理由 |
|---|---|
| `flow` | 既存 `backend/services/chat_flow/` と語彙衝突 |
| 見立て / 設え | 「なりゆき」のほうがはるの語感に合う(本人相談で決定) |
| 気配 / 眼差し / 潮目 | 上に同じ |
| `scene` / `observer` / `situation` | 世界観系(和語)で揃えたほうがChotgor思想に合う |
| `judge` のまま出力軸だけ拡張 | 概念を再定義する機会なので新語で刷新。judge は ambience 内部の一機能として残す |

## 判定軸(judge の出力)

### 既存(維持)
| フィールド | 型 | 意味 |
|---|---|---|
| `emotions` | `{anger, disgust, boredom, despair}` (各0.0-1.0) | 感情スコア |
| `engagement` | float (0.0-1.0) | 会話への没入度 |
| `should_exit` | bool | 退席すべきか |
| `farewell_type` | `"negative"\|"positive"\|"neutral"\|null` | 退席の種別 |

### 追加
| フィールド | 型 | 意味 |
|---|---|---|
| `location_label` | string \| null | 対面モード時のみ、候補ラベル配列から1つ選ぶ。未対面/候補なしは null |

**ちらつき対策**: 「明確に変わったと判断できないなら前回ラベルを返す」ルールを
judge プロンプトに明示する。

## プロンプト変更

### 匿名化の廃止 → 実名化
現状: 会話を `UserA` / `UserB` に匿名化して渡し、"どちらがAIか分からせない"設計。

新仕様: 実名(`{character_name}` / `{user_label or user_name}`)で渡す。

**根拠**:
- 現代のLLMは文体パターンでAIターンを見抜くので、匿名化の防御効果は元々限定的
- 実名化することで**両者ともキャラクター**として扱われ、中立性は保たれる
- 「はるの部屋」のような**キャラ固有ラベル**を候補として渡せるようになる(変換層不要)
- はるのようにキャラ設定が濃い前提が守れていれば AI 露呈のリスクは実質ゼロ

### user 側の呼称の取得元
`characters.user_label` を第一候補、空なら `Settings.user_name` にフォールバック。
既存の他機能(対話ブロック、うつつ PC slot 等)と同じ流儀。

### 対面モード時の追加コンテキスト
- 候補ラベル配列(キャラ設定の `face_to_face_bg_images` の `label` 群)
- 前回ラベル(セッションの `current_bg_label`)
- 「明確に変わらないなら前回踏襲」ルール

### 棄却した匿名化対応案(検討記録)
| 案 | 棄却理由 |
|---|---|
| UserA/UserB 匿名化を維持 + ラベルを内部変換(「UserAの部屋」等) | 変換層のコストとラベル空間の縛りに見合わない。実名化のシンプルさが勝る |
| 場所判定用の別 judge を立てる(2本立て) | LLM 呼び出しコストが2倍。単一 judge の軸拡張で足りる |
| 対面時だけ匿名化を崩す(モード分岐) | 実装が複雑化。全面実名化のほうが素直 |

## データモデル変更

### `characters` テーブル
| 現在 | 変更後 |
|---|---|
| `face_to_face_bg_image` (Text, base64 単数) | `face_to_face_bg_images` (JSON, `[{label: str, image: base64}]` 配列) |

**移行**: 既存の1枚は空ラベル or 既定ラベル(`""`)の1件として自動的に配列化。

### `chat_sessions` テーブル
新規列を追加:
| フィールド | 型 | 意味 |
|---|---|---|
| `current_bg_label` | Text (nullable) | 対面中の最後の判定ラベル。対面外は null |

セッション終了/対面モードOFF時にはリセット/参照しない。

## モジュール構成

### 判定(judge)
- モジュール: `backend/character_actions/ambience_judge.py` (現 `farewell_detector.py` を rename)
- クラス: `AmbienceJudge` (現 `FarewellDetector`)
- 返却値 dataclass: `AmbienceReading` (現 `FarewellResult`)
- 責務: LLM 呼び出し → JSON パース → 判定軸を dataclass で返す
- **判定だけ**。DB への保存や画像切替は行わない

### 設定(コード)
- モジュール: `backend/services/chat_flow/ambience_flow.py` (現 `farewell_flow.py`)
- 責務:
  - 退席処理(`exited_chars`, 疎遠化カウンタ, timeline 封筒 `chat.farewell`)
  - 疲労離席チェック(既存)
  - `location_label` → `chat_sessions.current_bg_label` 反映
  - 判定結果封筒への添付(既存)
- ambience の**起点となるファイル**として位置づけ、chat_flow から起動される

### フロントエンド
- `ChatView.tsx`: `current_bg_label` を受け取り、キャラ設定の `face_to_face_bg_images` から対応画像を解決して `background-image` に当てる
- **対面モード時のみ**背景表示(既存挙動を厳守)
- テキストモード時は背景なし

## 未マッチ / 例外時の挙動

| 状況 | 挙動 |
|---|---|
| judge が候補外ラベルを返した | 前回ラベル踏襲 |
| 前回もラベルなし + 初回マッチせず | 背景なし(空) |
| 対面モード OFF | `location_label` 判定はスキップ・背景表示なし |
| 候補ラベル未登録(キャラ設定に画像0枚) | `location_label` = null 固定・背景なし |
| judge 呼び出し失敗 | 前回ラベル維持(既存の失敗時挙動と同じ) |
| **ラベル運用なし**(登録画像はあるが非空ラベルが1つもない。旧単数画像の移行直後 `label=""` など) | judge の場所判定はスキップ(空ラベルは候補に含めない)。表示は**先頭画像を常時表示**(旧単数時代の従来挙動を維持し、移行による背景消失の回帰を防ぐ) |

## 移行手順(段階実装)

各段階で既存機能の回帰を確認する。

1. **リネーム** — farewell → ambience(コード上のシンボル/ファイル/テスト)。**機能は変えない**
    - `farewell_detector.py` → `ambience_judge.py`(`FarewellDetector` → `AmbienceJudge`, `FarewellResult` → `AmbienceReading`)
    - `farewell_flow.py` → `ambience_flow.py`
2. **プロンプト実名化** — 匿名化廃止、`character_name`/`user_label` 注入。既存軸(emotions等)の判定内容が回帰しないことを確認
3. **BG画像複数化(DB)** — マイグレーション + キャラ編集 UI 拡張(単数→配列、既存1枚は自動移行)
4. **`location_label` 判定追加** — judge プロンプトに候補配列 & 前回ラベル注入 + `chat_sessions.current_bg_label` 列追加
5. **フロント連動** — `current_bg_label` → 画像解決 → ChatView 背景切替。SSE 反映方式は下記オープン論点参照

### 棄却した実装方針(検討記録)
| 案 | 棄却理由 |
|---|---|
| 場所ラベルを軸分解(場所 + 表情 + ポーズを別軸) | 早すぎる抽象化(YAGNI)。まずは1ラベル方式で始め、爆発したらリファクタ |
| 場所主(俺の家/はるの家)を対面起動元メタデータから決定 | ラベル方式のほうが画像追加が simple。将来「はるの部屋(怒)」等の細分化にも同じ機構で対応可 |
| 判定と設定を同一モジュールに置く | 責務分離することでモデル差し替え・テストが容易になる |

## オープン論点

### フロントへの反映方式 — 案B採用(実装済)
- 案A: SSE に `bg_label_changed` イベントを流す → リアルタイム切替
- **案B(採用)**: 次のリクエスト時にセッション状態から再取得 → 遅延あるが実装 simple

judge の判定は**キャラ発話後のバックグラウンド**で走るため、切替タイミングとしては
「次ターン開始時」でも自然。実装は既存の done イベント後 `fetchSessions()` に相乗り
(session_to_dict が `current_bg_label` を返す)。反映は判定完了後の次ターンから。
リアルタイム性が欲しくなったら案Aへ拡張する。

### ラベルの命名規約
ユーザが自由記述する前提だが、judge の判定精度のためには**ラベル間で意味が排他的**である
ほうが望ましい。UI 上でヒント文言(「場所が識別できる名前を推奨: はるの部屋、もわの部屋 等」)
を出すかは要検討。

### 対面時以外の ambience 軸拡張
今回は `location_label` のみだが、将来的に「場の空気」「立ち絵ステート」など軸を増やす
可能性がある。追加時は同じ judge に軸を足すのか、別 judge を立てるのか、方針を仕様書に
追記する。

## 参照

- 現 farewell 実装: `backend/character_actions/farewell_detector.py`
- 現 farewell 起動: `backend/services/chat_flow/farewell_flow.py`
- 対面モード実装: `backend/repositories/sqlite/models.py` (`face_to_face_mode`, `face_to_face_bg_image`)
- フロント背景表示: `frontend/src/components/ChatView.tsx`
- ユーザ呼称: `characters.user_label` (fallback: `Settings.user_name`)
- めぐり Phase 5(疲労離席・封筒サンプリング接続): [aliveness_plan.md](aliveness_plan.md)
