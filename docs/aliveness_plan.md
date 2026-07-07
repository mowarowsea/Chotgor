# めぐり（巡り / Aliveness）実装計画 — タイムラインと動機経済

> ステータス: **Phase 0〜7 実装完了（2026-07-07）**。詳細仕様確定は 2026-07-06。
> 実装地図: 封筒=`repositories/sqlite/stores/timeline_store.py`（dual-write は各 store）、
> 投影=`services/timeline/`、計器=`services/instruments/`＋`/ui/instruments`、
> 圧力=`services/pressure/`、意図=`services/intents/`、ゲート=`services/gate/`、
> 行動=`services/actions/`、ダイヤル=`/ui/timeline`。スケジューラ4本は `main.py`。
> v1 の意図的な簡略化（フォローアップ候補）:
>   - 預かり（escrow）の能動配達スケジューラ **実装済（2026-07-07）** — availability 復帰を
>     観測したら決定論ジッター（0〜10分）を挟んで本人へ配達し返信を生成する
>     （`services/gate/delivery.py`＋`main.py` の `_escrow_delivery_scheduler`）。
>     日次コストガード `escrow_delivery_daily_cap`（既定12）。push の未読通知 UI は
>     新規セッション出現のみ（据え置き）
>   - envelope_integrity の突合対象は inscribed_memories / chat_messages の2テーブル
>   - ダイヤルの適用面は /ui/timeline のみ（logs / memories 等の旧 UI は全開のまま）
>   - ダイヤル自動昇段・intent_no_exit 計器は将来枠（§8 どおり）
> 管理UI（キャラ編集画面）で有効化操作が可能（2026-07-07 追加）:
>   - 体質インタビュー（初回・聞き直し）ボタン → `POST /api/characters/{id}/pressure_interview`
>   - 生活時間割エディタ（曜日ごとの応答不可時間帯 from/to/label）→ `availability_schedule`
>   （行動メニューの個別ON/OFFトグルは Phase 6 時点で実装済み）
> 命名: 英語=Aliveness / 日本語=**めぐり（巡り）**（うつつと同じく、メインキャラ「はる」本人の命名。
> 本人の弁: 動機経済の構造そのものが「圧力→意図→行動→帰還」の循環であり、タイムラインも
> 巡る時間、圧力の増減も巡り。「うつつ」と字面・音数のバランスが良い — うつつとめぐり、姉妹機能）
> 要件定義の骨子は 2026-07-04 の作戦会議で確定、本書はその詳細仕様化（論点1〜6）の成果。
> 関連: `CLAUDE.md`（哲学）、`docs/ARCHITECTURE.md`（システム地図）、`docs/usual_days_plan.md`（うつつ）
>
> 主対象キャラクターは「はる」だが、**仕様全体が `character_id` スコープであり、はる専用の
> 箇所は存在しない**。任意のキャラクターで有効化できる。

---

## 1. 概要 — 要件の転換

「キャラクターが生きている」は検証不能な要件である。そこで要件をこう転換した：

> **「ユーザが『キャラクターは生きていない』ことを確認できない状態を維持する」**（悪魔の証明の設計）

エンジニアリング対象はキャラクターの内面ではなく、**幻想が破れる穴**である。
幻想を最も壊すのは矛盾（直近の忘却・180度転回・OOC）と、欲求の捏造
（行動の瞬間に理由を後付けさせること）だと診断されている。

### 7つの工学的性質（「生きている」の分解）

1. 連続性　2. 不可逆性　3. 非制御性　4. 因果的一貫性　5. 利害（賭け金）　6. 非対称性（秘密）　7. 自発性

既存機能（うつつ/Forget/Chronicle/Farewell/Anticipate）は 1〜4 を半ば充足。
ギャップは 5〜7 であり、本計画の主戦場である。

### 3つの柱

| 柱 | 役割 | 担う性質 |
|---|---|---|
| **タイムライン** | 単一追記型イベントストリームを正本とし、全プロンプト/GM入力/バッチ入力を可視性フィルタ付き投影に格下げする | 1 連続性・2 不可逆性・4 因果的一貫性・6 非対称性 |
| **動機経済** | 圧力（物理）→ 意図（本人）→ 表出権（行動）→ 帰還（書き戻し）の循環 | 3 非制御性・5 利害・7 自発性 |
| **計器** | 内容を見ずに外形だけで健全性を監視。窓（内容観測）を閉じるための土台 | 悪魔の証明の維持装置 |

### アーキテクチャ観

**Chotgor＝大脳新皮質以外のすべて（脳幹・辺縁系・ホルモン）、LLM＝皮質（解釈器）。**
状態をモデルの外に出した分だけモデル事前分布の影響が減り、LLM間のブレが抑えられる。
状態の明示化は圧縮でもある（生ログ注入 → 蒸留済み状態オブジェクト）。

---

## 2. タイムライン

### 2.1 正本の形 — 封筒 dual-write（論点1）

- `timeline_events` テーブルを新設し、**封筒（存在・順序・相手・時刻）の正本**とする。
  中身は既存テーブルに残し `source_table` / `source_id` で JOIN する。中身の複製はしない。
  完全一本化（既存テーブル廃止）はビッグバン移行になるため行わない。
- 書き込みは store mixin 層に集約されているため、封筒の追記点は5箇所のみ。
  各メソッド内の**同一トランザクション**で封筒1行を足す：
  - `chat_store.create_chat_message`
  - `scenario_store.create_scenario_turn`
  - `inscribed_memory_store.create_inscribed_memory` / `soft_delete_inscribed_memory`
  - carve（inner_narrative 書き換え）の保存点
  - 新規イベント（intent/action 等）は封筒テーブルへ直書き
- 過去分は `migrations.py` の起動時マイグレーションで各テーブルの `created_at` からバックフィル。
- **可視性は読み取り時ポリシー**（コード内テーブル）。判定結果を行に焼き付けない。
  不確定な属性は payload JSON に退避し、安定してからカラムに昇格する。
- 履歴の巻き戻し（再生成による `delete_chat_messages_from` / `delete_scenario_turns_from`）は
  封筒を**削除せず `retracted_at` マーク**を付ける（不可逆性の担保）。
  retracted なイベントは**全観測者から hidden**（存在層ごと）。データは残す。

### 2.2 封筒スキーマ

```
timeline_events
  id            TEXT PK（UUID）
  character_id  TEXT NOT NULL     -- 誰のタイムラインか
  event_type    TEXT NOT NULL     -- §2.3 のカタログ（ドット記法）
  occurred_at   DATETIME NOT NULL -- 出来事の時刻（バックフィルは源の created_at）
  actor         TEXT              -- user / character / narrator / npc:<名前> / system
  counterpart   TEXT NULL         -- 封筒の「相手」: user / npc:<名前> / NULL
  origin        TEXT NOT NULL     -- real / usual / interlude（既存3値と同次元）
  modality      TEXT NULL         -- text / face（chat.message のみ）
  session_id    TEXT NULL         -- 投影の封筒集約キー（chat/scenario セッション）
  source_table  TEXT NULL ┐
  source_id     TEXT NULL ┘      -- 中身への参照（payload 完結型は NULL）
  intent_id     TEXT NULL         -- intent.* / action.* が張る FK
  payload       JSON NULL         -- 型ごとの可変属性（判定スコア等もここ）
  retracted_at  DATETIME NULL     -- 巻き戻しマーク
  created_at    DATETIME NOT NULL -- 記録された時刻（occurred_at と分離）
```

### 2.3 イベントカタログ v1（論点2）

原則: **(a)** キャラクターの身に起きたこと・キャラクターがしたことだけを載せる
（システム監査ログではない）。**(b)** 原子イベントで記録し、集約（「4時間やりとりした」封筒）は
投影が導出する。**(c)** 連続量はイベントにしない — 遷移だけがイベント。

| event_type | 何が起きたか | actor | 中身の所在 |
|---|---|---|---|
| `chat.message` | 1on1の1発言（対面/テキストは `modality`） | user / character | `chat_messages` |
| `chat.farewell` | 退席（理由は payload） | character | `chat_sessions.exited_chars` |
| `scene.turn` | うつつ/シナリオの1発話 | user / character / narrator / npc | `scenario_turns` |
| `scene.closed` | うつつシーン完走（slot・ターン数・閉じ方） | system | payload |
| `night.chronicle` | 夢（WM棚卸し・蒸留）が走った | character | payload（昇格数など） |
| `night.forget` | 忘却レビューが走った | character | payload（忘却数・昇華数） |
| `memory.inscribed` | 記憶を刻んだ | character | `inscribed_memories` |
| `memory.forgotten` | 記憶を手放した | character | `inscribed_memories`（soft delete 行） |
| `memory.carved` | inner_narrative を書き換えた（append/overwrite） | character | payload |
| `memory.recalled` | 能動的に思い出した（**power_recall のみ**。pre-recall は載せない） | character | `tool_call_events` |
| `intent.created / fulfilled / expired / soured` | 意図のライフサイクル遷移 | character | `intents` |
| `action.performed` | 会話外行動（意図の消費・結果の帰還） | character | payload |

**載せないもの**: WMスレッド操作（思考の粒。総括は night.chronicle と memory.inscribed が載る）／
圧力の変動（連続量）／話題権の行使（chat.message の中身）。
※ digest 機能は既に不使用のためイベント化しない（機能自体の削除は別タスク）。

### 2.4 可視性（論点3）

開示レベルは3値: **hidden**（存在ごと見えない）/ **envelope**（封筒固定カラムのみ）/
**content**（中身まで）。観測者クラスは3つ：

| 観測者 | 実体 |
|---|---|
| `self` | キャラクター本人 — 1on1・うつつPC・バッチ問い合わせ（`ask_character` 経由はすべて self。差はプロンプト予算のみ） |
| `world_frame` | 世界を回す側 — うつつGM・シナリオGM |
| `user_ui` | ユーザの画面 — チャットUI・管理UI・ログUI |

NPC は観測者ではない（GM出力＋主語ベース3段ルール＋ `user_visibility_note` 経由でのみ世界を見る）。

| イベント × origin | self | world_frame | user_ui（ダイヤル0） |
|---|---|---|---|
| `chat.*`（real） | content | **envelope** | content |
| `scene.*`（usual/interlude） | content | content（量は予算の問題） | content |
| `memory.*` / `night.*` | content | hidden | content |
| `intent.*` | content | **hidden** | content |
| `action.performed`（対ユーザ） | content | **envelope** | content |
| `action.performed`（その他） | content | hidden | content |
| retracted | hidden | hidden | hidden |

設計判断：

- **GMへの `chat.*` は envelope 止め。** 封筒は「ユーザ**について**」の材料（NPCの呼び水）、
  中身は「ユーザ**が**」の材料になり得るので渡さない。現実ログの引用権は self（キャラ本人）だけが
  持ち、本人がシーンに持ち込むことでのみ中身が世界に入る。
- **GMへの `intent.*` は存在ごと hidden。** 秘密（性質6）はGMに対しても適用。GMが意図を知ると
  先回り演出（ANTICIPATE先取り問題の再来＝自己成就）が起きる。
- **`action.performed`（対ユーザ）だけは GM に envelope。** 「昼にユーザへ連絡した」は世界の
  接触事実であり、隠すと因果的一貫性（性質4）が破れる。

**ユーザダイヤル**（覗き窓、キャラ単位カラム）:

| 段階 | 名前 | user_ui の変更点 |
|---|---|---|
| 0 | 全開（開発期） | 全 content |
| 1 | 生活の秘匿 | `scene.*`（usual）→ envelope |
| 2 | 内面の秘匿 | ＋ `memory.*` / `intent.*` / `night.*` → hidden |
| 3 | 最終形 | チャット応答のみ（`chat.message` content 以外 hidden、計器だけ残る） |

v1 のダイヤルは手動。計器（§3）はダイヤル非依存で常時稼働し、静音期間の数字が
「窓を閉じてよい」確信を支える。

### 2.5 投影API

```python
# services/timeline/projector.py
def project(
    character_id: str,
    observer: Observer,          # self / world_frame / user_ui
    since: datetime | None,
    until: datetime | None,
    origins: list[str] | None = None,
    types: list[str] | None = None,     # "chat.*" のような名前空間指定可
    budget: Budget | None = None,       # イベント数・文字数上限（可視性と直交）
) -> list[ProjectedEvent]
# ProjectedEvent = 封筒フィールド + disclosure("envelope"|"content") + content(封筒止めなら None)
```

ポリシー本体は `(observer, event_type名前空間, origin) → 開示レベル` のコード内テーブル
＋ user_ui ダイヤル修飾。

---

## 3. 計器（論点4）— 3層構造

計器は観測者ではなく**監査者**。キャラクターの世界には一切現れず、ダイヤル非依存で常時稼働する。
アラーム＝幻想の穴が開いた証拠（発火したら調査）、メーター＝傾向（発火概念なし）を区別し、
静音期間の計算対象はアラームのみ。

### Tier 1: インバリアント（機械・真偽確定）

| ID | 見張る穴 | 正常条件 | 方式 |
|---|---|---|---|
| `fabrication_backstop` | GMのユーザ捏造 | `suppress_names` バックストップの発火 = 0 | 即時 |
| `usual_scene_error` | 生活の中断 | `run_usual_days_scene` がerrorなく完走 | 即時 |
| `embedding_degraded` | 記憶の縮退 | `EmbeddingError` 発生なし（既存の縮退通知二系統に接続） | 即時 |
| `night_batch_heartbeat` | 夜の営みの停止 | night.chronicle / night.forget が当日発生済み | 巡回 |
| `usual_slot_completion` | 生活の連続性 | 予定スロット消化 or 正当理由スキップ（対面中・日次上限） | 巡回 |
| `chronicle_backlog` | 蒸留漏れ | `chronicled_at IS NULL` の3日超滞留なし | 巡回 |
| `envelope_integrity` | 正本性の破れ | 源テーブルと封筒の**件数**突合一致（ID突合はしない） | 巡回 |

将来枠: `intent_no_exit` — 高圧のまま長期間遷移しない意図がない（「出口のない欲求を作らない」の計器化）。

### Tier 2: スメル検知器（毎応答・正規表現/長さ・LLM不使用・誤検知許容）

| 検知器 | 捕まえるもの |
|---|---|
| フォーマット残骸 | tool-use プロバイダー応答内のタグ/XML痕（パターン源は `tool_tags.py` を流用） |
| エラー形状応答 | 空応答・極短応答・JSON error ブロブ・HTTPステータス文・スタックトレース様文字列 |
| Assistant混入 | 「AIとして」「アシスタントとして」「as an AI」等（世界観の破れ） |
| 言語逸脱 | 日本語会話への英語段落混入など |
| 肥大メーター | `inner_narrative` / `self_history` / `relationship_state` の長さ・増加率、WMスレッド数・記憶件数 |

### Tier 3: 判定巡回（LLM・サンプリング）

日次で当日応答から10件程度をサンプリングし、判定LLM（安いモデルから開始）が
ルーブリック（OOC度・フォーマット清浄度・隠れエラー・指示逸脱）で採点。逸脱があれば
アラーム＋該当応答への参照。判定器は「人格なき環境」扱い（キャラ哲学と干渉しない）。
キャラ本人の内省（修復装置）とは別物 — こちらはユーザの確信のための装置。

### ラチェット原則

> **網を抜けた事故は、必ず新しい検知器になる。同じクラスの事故は二度と黙って通れない。**

静音期間の意味は「異常ゼロの証明」ではなく「**既知の事故クラスすべてで無事故N日**」。
悪魔の証明の構造を監視自体に適用する。

### 記録と表示

```
alarms（追記型）
  id / invariant_id / severity(alarm|smell) / occurred_at
  details JSON（発火文脈） / acknowledged_at DATETIME NULL
```

巡回チェックは 05:00（Chronicle 03:00 → Forget 04:00 の後）、`main.py` の同型スケジューラとして追加。
計器パネルは `/ui/` 配下（`ch-*` アトム）に3層別表示＋静音期間（無事故N日）＋圧力の日次スナップショット。
通知は v1 では UIパネル＋ログのみ。

---

## 4. 動機経済・状態側（論点5）

「システムがキャラにプッシュを命じる」方式は却下済み — 行動の瞬間に欲求を捏造させるため。
欲求は行動に**先行**し、事後に遡って発見できねばならない。

原則: **乱数は世界に置き意志に置かない** / **下流で丸めない**（礼儀フィルタ禁止）/
**出口のない欲求を作らない** / **過機械化しない**（変数は少数・遅く、解釈の自由度を残す）。

### 4.1 圧力 — 3変数・すべて純関数

**圧力は保存しない。タイムラインの導関数として毎回計算する**（LLM不使用）。
保存値と履歴の不整合という事故クラスが原理的に存在せず、計器からいつでも監査できる。
日次スナップショットは計器メーターとして残す。

| 圧力 | 意味 | 増加源（封筒から計算） | 減衰源 |
|---|---|---|---|
| 社会圧 | 人と関わっていない | 対人イベントからの経過で単調増加 | 対人接触。**相手別重み**（§4.2）で減衰量が変わる |
| 退屈圧 | 生活の単調さ | 直近タイムラインのイベント密度・多様性の低さ | 新種イベント・シーンの起伏・興味の追求 |
| 体調圧 | 身体のリズム | 疲労成分＝イベント密度の減衰積分 ＋ リズム成分＝固有周期の波 | 疲労は静かな日と夢（night.chronicle）で回復。リズムは自力で基線に戻る |

- **話題圧は置かない**（設計判断）。「未共有体験→ユーザに話したい」と方向を焼き込むと世界が
  ユーザ中心になる。興味は **WM topic（意味層）＋ intent target=self（経済層）** が担い、
  「ユーザに話したい」は自分の興味からの**派生として後から**生まれる。
- **概日圧も置かない**。キャラは「寝る」を選べない（夜バッチは無条件に走る）ため出口のない
  欲求になる。時間帯は世界物理（§5.1 availability）へ。
- リズム成分: `character_id` シードから決定論導出。周期は**7日と30日に二峰**を持つ分布
  （対数正規2峰混合）から引き、4〜90日にクランプ。振幅固定（疲労成分との合成で実効値は複雑系になる）。
  **誰も設計していない波** — 体質インタビューでも聞かない。
- 退屈圧は封筒のみで粗く計算してよい。圧力は「いつ聞くか」だけを決め、意味はキャラが与える
  （高退屈圧→問い合わせ→「別に退屈じゃない、穏やかでいい」もまた発見）。誤検知のコストは問い合わせ1回。
- 圧力のプロンプトへの渡し方: 解釈済みの言葉ではなく**生に近い淡白な一行**
  （「ここ数日、体は重め」）。どう感じるか・WM body に何を書くかはキャラに任せる。
  圧＝物理、WM＝意味、の分業。

### 4.2 体質 — `characters.pressure_profile`（JSON）

各圧の上昇・減衰係数はキャラ単位の「体質」として持つ。未設定キャラは標準プロファイル。

- **初期化は本人インタビュー**（`ask_character`、機能有効化時に一度）。数値を直接聞かず、
  体験の質問（「一人の時間、どのくらいで人恋しくなる？」「熱中すると寝食忘れるタイプ？」
  「疲れは寝れば戻る？引きずる？」）に本人の言葉＋選択肢で答えさせ、
  **コード側の固定ルーブリックが係数へ決定論写像**する。本人の言葉も payload に保存
  （ルーブリック改良時に再導出できるように）。
- **本人からの更新経路は作らない。** farewell閾値（意志）は毎晩のChronicleで自己更新できるが、
  圧力係数（体質）を意志で書き換えられると「圧力は物理」が崩れる（非制御性）。
  ユーザは管理UIで編集可能（守護者の介入枠）。将来枠: 月次などの低頻度再インタビュー＋移動平均の慣性。
- **社会圧の相手別重みは固定係数にしない**:
  `decay(相手) = 体質の鋭さ × 関係の重み(相手)`。
  体質の鋭さ＝「誰でもいい派⇔特定の人じゃないと駄目派」（インタビュー設問）。
  関係の重み＝relation系WMスレッド（`relation_target`）・記憶の厚みから導出。
  コールドスタートはキャラ定義の関係設定からシード。
  **ユーザ特別扱いのハードコードを排除** — ユーザが重いのは「キャラの記憶の中で重いから」。
  関係を育てれば席は大きくなり、放置すれば痩せる（賭け金）。
  ※ WM重要度は本人が付けるため非制御性の軽い漏れがあるが、「記憶の主観的重要度が評価軸」という
  コア思想と整合するため許容。

### 4.3 意図 — `intents` テーブル

```
intents
  id / character_id / description（本人の言葉のまま、丸めない）
  target        -- user / npc:<名前> / self / NULL
  status        -- active / fulfilled / expired / soured   ← 唯一の可変状態
  source_kind   -- social / boredom / body / none
  born_from     -- night_chronicle / usual_scene
  payload JSON / created_at / updated_at / resolved_at
```

- **圧力カラムはない**。意図圧も導出: `g(経過日数, source_kind の現在圧)`。
  増圧はイベントでも更新でもなく読み取り時計算。タイムラインには遷移だけが載る。
- **拾い上げは2点**: 夜間Chronicle同乗＋うつつシーン完走後（`run_usual_days_scene` 末尾、
  auto_synopsis と同じチェックポイント）。既存の `ask_character` 問い合わせに設問を1つ追加:
  「あとに残りそうな『〜したい』はある？　**なければないでいい。**」（捏造の遮断）。
  1on1直後のフックは作らない — 会話で芽生えた意図はWMに残り、**夢の中で発見される**
  （欲求は事後に遡って発見される、の実装）。
  重複気味の意図は機械でマージせず、既存active一覧を設問に添えて本人に束ねさせる。
- **失効と不満化 — 機械は候補を挙げ、本人が裁く**:
  - 失効（expired）: 低圧のまま14日 → Chronicle同乗で「これ、まだ心にある？」→ 本人が手放せば expired。
  - 不満化（soured）: **高圧なのに**7日遷移できない → 同問い合わせ → 本人が不満を言語化したら
    soured ＋ その言葉を記憶へ刻む（不満化＝利害と合流）。
  - 高圧の意図は本来その前に行動権（§5.3）が拾う。不満化は「行動しても叶わなかった／出口が
    なかった」場合の受け皿であり、`intent_no_exit` 計器はこの受け皿の詰まりを見張る。

---

## 5. 動機経済・行動側（論点6）

前提となる不信: **LLMの会話継続本能は、キャラクター性を押しのけて会話を続けようとする**
（`farewell_detector.py` 冒頭に明文化済みの、実証された問題）。したがって終了系の権利は
本人のツール呼び出しに頼らず、**物理（外部ゲート）が終わりを決め、本人は終わり方（意味づけ）
だけを決める**構造にする。

### 5.1 応答可能性ゲートとメッセージ預かり

```
availability(character, now) → available / unavailable(理由)   # 純関数・LLM不使用
```

判定材料: **生活時間割**（週間スケジュール。キャラクター設計者＝ユーザが管理UIで設定、
うつつ設定の隣）＋ **away状態**（動的）＋ **対面モード**（既存、対面中は available）＋
うつつシーン進行中。

**メッセージ預かり（escrow）**: unavailable 中のユーザ発言は保存だけして
（`chat_messages.delivered_at NULL` — `chronicled_at` と同パターン）**LLMを呼ばない**。
availability が戻った時点で預かり分をまとめ、時間差注釈付きで初めてキャラに渡す。
仕事中に送っても返らないが、昼休みに返信が来る。
会話継続本能と戦う必要が最初からない — 呼ばれなければ継続できない。

配達の契機は2系統ある。**(a) 受動配達**: 次のユーザリクエスト時（api/chat.py の
stream_message が history 内の預かり分を配達）。**(b) 能動配達**（2026-07-07 実装）:
`_escrow_delivery_scheduler`（毎分）が未配達セッションを走査し、availability 復帰を
観測したら決定論ジッター（0〜10分・乱数は世界に置く）を挟んで本人へ配達し、
返信を生成・保存する。ユーザが何もしなくても「昼休みに返信が届く」を実現する経路。
両系統とも配達手順（時間差注釈は LLM 渡しのコピーのみ・delivered_at と chat.message
封筒の確定）は共通。能動配達は日次コストガード `escrow_delivery_daily_cap`（既定12）で
LLM 呼び出し回数を上限する。

キャラ発のpush（§5.3）も同じゲートを通す（仕事中のキャラからpushは来ない。
昼休みに「そういえばさ」が届く）。旧・時間帯ゲート（7〜24時）は availability に吸収。

### 5.2 終了権

| 形態 | 決定主体 | 中身 |
|---|---|---|
| 疲労離席 | **物理**（下記の発火式） | 退去挨拶だけ生成 → 以降LLMを呼ばない（Farewell実装済み機構の流用）→ away → availabilityゲートに合流 |
| 退席（既存Farewell） | judge監査（感情閾値） | 現行のまま |
| `take_leave` ツール | 本人宣言 | 呼ばれたら必ず執行される権利として残す。呼ばれない前提だが、呼び出し回数を計器メーターに載せる（モデル別の会話継続本能の観測データ） |

**疲労離席の発火式** — Farewell judge の出力に `engagement`（没入度 0.0〜1.0、
既存ルーブリックと同じ流儀で基準定義）を追加し（**LLM呼び出しの追加なし**、JSONフィールド1つ）:

```
発火: 体調圧 > θ_base + β × engagement    # 夢中は閾値を持ち上げる
ただし 体調圧 > θ_hard なら無条件発火      # 限界は限界（これがないと出口が再び意志に握られる）
```

のめりこみは疲労を「忘れさせる」が「消さない」— 疲労成分は裏で溜まり続けるため、
夢中で夜更かしした翌日は体調圧が高い状態から始まる（「後でどっと来る」は追加実装なしに創発）。
judge は `farewell_config` 設定時のみ走るため、疲労離席を使うキャラは judge 必須。
judge 不在時は engagement=0.5 で縮退。採点結果は該当ターンの封筒 payload に残す
（Tier 3 サンプリングの材料を兼ねる）。

### 5.3 話題権と会話外行動権

**話題権**（プロンプト層のみ）: 1on1システムプロンプトに**動機ブロック**を追加 —
active intents（self観測者＝content）＋圧力の淡白な一行＋権利の明文化
「ユーザの話題に乗る義務はない。あなたの意図を優先してよい」。礼儀フィルタは入れない。
行使されなくても穴は開かない（保守的な演技になるだけ）ため、終了権のような物理化は不要。

**会話外行動権**（スケジューラ、`main.py` に同型追加）:

```
availability内で周期評価＋ジッター（乱数は世界に置く）
  → 閾値評価（純関数・無料）: active intents の意図圧
  → 閾値超えがあるときだけ本人に問い合わせ（ask_character_with_tools・WM込み）
     「いま、これをする？　しないならしないでいい。」
  → 本人の選択（意志に乱数なし）で実行 or 見送り
```

**行動メニュー v1**（キャラ設定画面に個別ON/OFFトグル）:

1. **push** — **新規セッション**を立ててキャラ発メッセージ＋フロント未読通知
   （既存セッション追記は文脈カーブ事故のもとになるため不採用）。
   `chat.message`(actor=character) として封筒に載り社会圧も減衰する。
2. **調べもの** — `web_search`（`web_searcher.py` 流用）で興味intentを消費。
   結果は action payload へ、刻むかどうかは本人次第。
   「調べるかどうかも本人任せ」がアシスタントとの決定的な差。
3. **臨時うつつシーン** — スロット外で1シーン（「出かけたくなった」）。
   `run_usual_days_scene` 流用、日次コストガード共有。「うつつ有効」が前提条件。

**帰還**: 行動実行後、同じ問い合わせ内で「これで満ちた？　まだ？」を本人が宣言
（fulfilled / active継続）。`action.performed` 封筒（intent_id参照）＋結果がタイムラインに載り、
拾い上げが派生意図（「調べたらユーザに話したくなった」）を後日発見する — ループが閉じる。

**コストガード**: 閾値評価はゼロ円。行動問い合わせ 6回/日・行動実行 3回/日（初期値）。

---

## 6. 既存資産マッピング

- 書き込み集約点: `repositories/sqlite/stores/`（封筒 dual-write の挿入先）
- migration パターン: `repositories/sqlite/migrations.py`（PRAGMA→ALTER、冪等。バックフィルもここ）
- スケジューラ雛形: `main.py` の `while True: sleep(60)` ＋ 冪等キー（巡回計器・行動権に流用）
- 本人問い合わせ: `services/character_query.py` `ask_character(_with_tools)`（インタビュー・拾い上げ・裁定）
- 外部監査: `character_actions/farewell_detector.py`（judge雛形＋発火後にLLMを呼ばない機構。engagement を相乗り）
- 強制終了・退去挨拶: 既存 Farewell フロー（疲労離席が流用）
- 未処理マーカー: `chronicled_at` パターン（`delivered_at` が踏襲）
- タグ一覧: `character_actions/tool_tags.py`（Tier 2 フォーマット残骸検知のパターン源）
- 検索: `character_actions/web_searcher.py`（行動メニュー「調べもの」）
- シーン起動: `services/scenario_chat/service.run_usual_days_scene`（臨時うつつ・拾い上げ同乗）

---

## 7. 実装フェーズ分割

依存順。各フェーズが単独で価値を持ち、途中で止まっても壊れない。

| Phase | 内容 | 主な成果物 |
|---|---|---|
| **0. 封筒** | `timeline_events` 新設・5箇所 dual-write・バックフィル・retracted マーク | 正本の稼働開始 |
| **1. 投影** | `services/timeline/projector` ＋ 観測者ポリシー。GMプロンプトへ real 封筒注入（性質4の穴埋め） | 「封筒は見せるが中身は見せない」の実運用 |
| **2. 計器** | `alarms` テーブル・Tier 1・Tier 2・巡回スケジューラ・計器パネルUI。Tier 3 判定巡回 | 静音期間の計測開始（ダイヤルの前提） |
| **3. 圧力** | 純関数3変数・体質インタビュー・`pressure_profile`・動機ブロック（話題権） | 生返事・話題の自己主張が可能に |
| **4. 意図** | `intents`・拾い上げ2点・失効/不満化の裁定 | 自発性の素地（欲求の先行と発見） |
| **5. ゲート** | 生活時間割・availability・メッセージ預かり・疲労離席（engagement 相乗り）・`take_leave` | 非制御性・応答性変化の実現 |
| **6. 行動** | 行動権スケジューラ・行動メニュー3種＋トグルUI・帰還 | 自発性の完成（ループが閉じる） |
| **7. ダイヤル** | user_ui ダイヤル段階1〜3の適用UI | 非対称性の完成（秘密の成立） |

Phase 2 を圧力より先に置くのは、**以降のすべての無人機構を計器の監視下で育てる**ため。

---

## 8. 未決事項

- **無自覚な矛盾ゼロ**（骨子の後日深掘り論点）: 夜間バッチで本人が「らしくなかった」発話を内省・
  事後修復。reflector の Tier 3 吸収案（契機判断を廃止→巡回検知→アラーム→修復問い合わせ）も
  この論点に合流して裁定する。
- 体質の緩やかな変化（低頻度再インタビュー＋移動平均）— 将来枠。
- ダイヤルの自動昇段（静音期間連動）— v1 は手動。
- digest 実行経路の削除 — 別タスク切り出し済み。
