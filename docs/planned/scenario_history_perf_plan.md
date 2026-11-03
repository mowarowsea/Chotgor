# シナリオ履歴の転送・描画コスト削減

> ステータス: **実装完了（2026-07-31, `c242abb` ＋ 無限スクロール `b0a11cd`）**。
> 追補: 1on1 への履歴ウィンドウ横展開を実装（2026-11-03）。内訳は末尾「1on1 への横展開」。

## 目的

長寿命のシナリオセッション（実測 854 ターン）で、次の3つの体感遅延を解消する。

1. レスポンス完了後、所要時間が出たあとも「GM が考えています…」が 10 秒ほど残る
2. 枝ナビ（◀ 2/3 ▶）の切り替えが重い
3. セッションを開くときのもたつき

いずれも LLM 呼び出しの遅さではなく、**全ターンを毎回転送し・毎回作り直している**ことが原因。

## 現状の実測（session=854 ターン）

`GET /api/scenario_chat/sessions/{sid}/turns` のペイロード内訳（UTF-8 で 2.65MB）:

| フィールド | 比率 |
|---|---|
| `raw_response` | **86.0%** |
| `content` | 8.7% |
| `anticipation` | 1.3% |
| その他（id / session_id / variant_siblings / created_at 等） | 4% |

サーバ側の処理時間は 0.11 秒（＝ボトルネックは DB ではない）。残りはネットワーク転送・
`JSON.parse`・854 バブルぶんの React 再構築が占める。

さらにフロントは、以下の3経路すべてで**全ターンを取り直して丸ごと差し替えて**いた。

- レスポンス完了直後（`handleScenarioSend` の後処理）
- 枝切り替え（`POST /turns/activate` の戻り値）
- セッション選択・シナリオ起動

## 採用する設計

### 1. `raw_response` を指紋（`response_key`）に置き換える

`raw_response` は「モデルが実際に何を出したか」のデバッグ記録であり、**フロントは一度も
表示に使っていない**。用途は等値比較だけ:

- レスポンスグループの畳み込み（`ScenarioChatView/index.tsx` の `gmGroupTailById`）
- 再生成・破棄の対象となる末尾レスポンス列の先頭探索（`useScenarioChat`）

したがって API レスポンスからは本文を落とし、**同一性だけを保つ短い指紋**を返す。

```
response_key = sha1(raw_response.encode("utf-8")).hexdigest()[:16]   # None は None のまま
```

- DB カラム `scenario_turns.raw_response` は**残す**（デバッグ記録としての価値は変わらない）。
  落とすのは API レスポンスだけ。
- `None` と `""` は指紋でも別値になるため、既存の比較セマンティクスがそのまま保たれる。
- これだけで 2.65MB → 約 370KB（1/7）になる。全経路に効く。

**衝突について**: 別々の LLM 呼び出しがまったく同一のテキストを返すと同じ指紋になり、
連続していれば 1 グループに畳まれる。ただしこれは `raw_response` 全文で比較していた
現状でも同じ挙動であり、退行ではない。

### 2. レスポンス完了時点で `sending` を落とす

`turn_complete`（＝GM/PC のレスポンス連鎖が終わり、ユーザ入力待ちに戻った合図）を
受けた時点で `sending` を false にする。従来は、その後に走る整合性の再取得
（`GET /turns` の全件差し替え）が終わるまで `sending` が true のままで、
`sending && pendingBubbles.length === 0` を条件とする「GM が考えています…」が
**バブルを出し終えたあとに再点灯**していた。

後処理（turns 再取得・セッション再取得）はそのまま裏で走らせる。ただし
`sending` を早く落とすとユーザが即座に次を送れるため、**送信の世代カウンタ**
（`sendSeqRef`）を導入し、古い送信の後処理が新しい送信の結果を上書きしないようにする。

### 3. 履歴のウィンドウ表示（直近 N 件 + 遡り読み込み）

854 バブルを常時 DOM に置くのをやめ、**直近 `TURN_PAGE_SIZE` 件**だけ取得・描画する。
上端へ近づくと 1 ページぶん過去へ自動で伸びる（**無限スクロール**）。

- API: `GET /turns?limit=N&before_index=M`
  - `before_index` 省略で末尾から `N` 件、指定時は `turn_index < M` の直近 `N` 件。
  - どちらも**昇順の配列**を返す（レスポンス形状は変えない）。
  - `limit` 省略時は従来どおり全件（既存呼び出し・テストの互換維持）。
- 「まだ上にあるか」は `取得件数 === limit` で近似する。ちょうど一致したときに
  空振りの追加取得が 1 回起きるが実害はない。
- `POST /turns/activate` も `?limit=N` を受け、切替後の本線を同じウィンドウで返す
  （枝切り替えを 1 往復のまま維持するため）。

**発火**: 履歴リストの先頭にセンチネル要素を置き、`IntersectionObserver`（root=スクロール
コンテナ、`rootMargin` で上方向に先読みマージン）で可視化を検知して次ページを取る。
`IntersectionObserver` は**交差状態が変化したときしか呼ばれない**ため、1 ページ足しても
センチネルがまだ可視域にある場合は再発火しない。読み込み完了（`loadingOlderTurns` が
false へ戻る）を effect の依存に入れて**購読し直す**ことで、その場合も継続して伸びる。

**スクロール制御**: 最下端への自動追従は「末尾ターン ID が変わったとき」だけに限定する。
遡り読み込みでは末尾が変わらないので発火しない。加えて、発火時点の「下端からの距離」
（`scrollHeight - scrollTop`）を控えておき、DOM 反映直後（`useLayoutEffect`）に同じ距離へ
戻して、先頭に積まれたぶんのずれを打ち消す。

**エクスポート**: 表示中の `turns` をそのまま渡すとウィンドウ内しか出力されないため、
エクスポートダイアログを開くときに**全件を取り直す**（`limit` なしの `GET /turns`）。

**再取得時のマージ**: レスポンス完了後の整合性再取得で丸ごと差し替えると、遡って
読み込んだ履歴が送信のたびに畳まれてしまう。取得ウィンドウの先頭 `turn_index` を境に、
それより古い表示中のターンは残してマージする（更新が要るのは末尾側だけ）。
枝の切替は分岐点より後が巻き戻り前方の並びも変わりうるため、そちらはウィンドウごと
リセットする。

## 検討したが採用しなかった案

- **仮想スクロール（react-window 等）の導入** — 依存を増やす。ウィンドウ + 遡り読み込みで
  実用上の DOM 数は十分抑えられ、転送量削減（案1・案3）と違って通信コストには効かない。
- **「以前のやり取りを読み込む」ボタン** — 初版はこれで実装したが、遡るたびにクリックが
  要るのが煩わしいため無限スクロールへ差し替えた。ボタンは残していない（自動読み込みが
  効いている間は押す機会がなく、二重の導線になるため）。
- **`onScroll` で `scrollTop` の閾値を見る** — ヘッダー表示判定の既存ハンドラに相乗りできるが、
  スクロールイベントが来ない状況（コンテンツ縮小・プログラムによる位置変更）で取りこぼす。
  `IntersectionObserver` なら状態変化そのものを見られる。
- **`generation_id` でレスポンスグループを判定する** — 1 リクエスト内で GM が複数回
  呼ばれる（`max_responses` までループする）ため粒度が粗すぎ、複数レスポンスが 1 つに
  畳まれてしまう。指紋の新設が必要。
- **`raw_response` を API から単純に削除する** — グルーピングとレスポンス境界探索が
  壊れる。同一性の情報だけは残す必要がある。
- **`turn_complete` の到達をもってサーバ側で `done` を送る（後処理をなくす）** —
  完了後の再取得は「ストリーム中に取りこぼした確定値をサーバの真値へ寄せる」役割があり、
  消すと整合性が落ちる。裏で走らせるだけにとどめる。
- **レスポンス形状を `{turns, has_more}` に変える** — 既存の呼び出し・テストを広く壊す。
  件数比較で代替できるので配列のままとする。

## 変更ファイル

| ファイル | 変更 |
|---------|------|
| `services/scenario_chat/serializers.py` | `scenario_turn_to_dict` が `raw_response` → `response_key`（指紋）を返す |
| `repositories/sqlite/stores/scenario_store.py` | `list_scenario_turns` に `limit` / `before_index` |
| `api/scenario_chat/sessions.py` | `GET /turns` と `POST /turns/activate` に `limit` / `before_index` クエリ |
| `frontend/src/api/scenario.ts` | `ScenarioTurn.raw_response` → `response_key`、取得系にページング引数 |
| `frontend/src/hooks/useScenarioChat.ts` | 指紋比較へ差し替え・`sending` 早落とし・世代カウンタ・遡り読み込み |
| `frontend/src/components/ScenarioChatView/index.tsx` | 指紋比較・遡りセンチネル（無限スクロール）・スクロール制御 |
| `frontend/src/App.tsx` | エクスポート時の全件取得 |

## 1on1 への横展開（2026-11-03 実装）

**方針は決定済み — 「共通化する。個別実装はなるべく薄く」**（2026-07-31 合意）。
1on1 にも同じウィンドウ + 無限スクロールを入れ、スクロール機構は両者で共有した。

### 現状の実測（1on1・着手前）

最大セッションで 210 メッセージ / 約 750KB。シナリオ（1098 ターン / 約 2.9MB）ほど
逼迫していないが、内訳の構造は同じで **`reasoning` がペイロードの 80〜92%** を占める。

### 共有したもの / しなかったもの

- **共有（新規フック `hooks/useInfiniteScrollTop.ts`）**: センチネル + `IntersectionObserver`
  （root=スクロールコンテナ・上方向 600px 先読み）+ 前方追加時のアンカー復元
  （発火時の「下端からの距離」を控えて `useLayoutEffect` で戻す）+ `loading` 依存の再購読。
  `ScenarioChatView` から抽出し、`MessageList`（1on1）と両方がこれを呼ぶ。
  「末尾 ID が変わったときだけ最下端追従」は各画面の既存 effect がそのまま担う
  （1on1 の `MessageList` は既に末尾 id 依存だったので、前方追加で飛ぶ問題は起きない）。
- **共有しない（取得層）**: 1on1 は `hooks/useMessageWindow.ts`、シナリオは
  `useScenarioChat` の `applyTurnWindow` / `mergeTurnWindow` / `loadOlderScenarioTurns`。
  カーソルも API も違うため、共通化すると器だけ増える。

### 着手前に決めた 4 点の裁定

1. **エンドポイント** — 両方やった。`GET /api/chat/sessions/{id}?limit=N` が末尾 N 件を
   `messages` に載せて `has_older_messages` を返す（セッションを開く 1 往復のまま）。
   遡りは新設の `GET /api/chat/sessions/{id}/messages?before=<message_id>&limit=N` →
   `{messages, has_older}`。`limit` 省略時の詳細は従来どおり全件（エクスポート・既存テスト互換）。
   「まだ上があるか」はシナリオの件数一致の近似ではなく **N+1 件取って 1 件余るか** で厳密に出す
   （`api/chat.py _list_message_window`）。
2. **カーソル** — `(created_at, rowid)` の複合キー。`chat_messages` に `turn_index` 相当の
   単調増加列は無く、`created_at` 単独では同時刻の衝突で境界が曖昧になる（`datetime.now()`
   既定で、user 発話と預かり通知のように続けて insert する行は Windows ＋ Python 3.12 以前の
   約 15ms 分解能では同時刻になる）。`list_chat_messages(session_id, limit=, before_id=)` が
   複合キーで `<` を切り、並び順も同じ複合キー（`chat_store._message_order`）に固定した。
   `_tail_after_anchor`（最新ターン判定）と `delete_chat_messages_from`（以降削除）も同じ順を
   使い、ウィンドウと境界が食い違わないようにしている。
   枝（variant）の復元は `created_at` を元の値のまま戻すので、同時刻の行が他に無い限り
   位置は変わらない（rowid 単独なら再 INSERT で末尾へ飛ぶが、第 1 キーが created_at なので
   その問題は起きない）。
   棄却: `(created_at, id)`（レビューで差し戻し。id は乱数 UUID なので同時刻の行が挿入順と
   無関係に並び、通知が発話の前に来る）、rowid 単独（枝の復元で順序が created_at と食い違う）、
   offset（送信で末尾が伸びるたびにずれる）。
   ウィンドウがキャラ発話だけで埋まったとき（能動配達の連打等）は、枝ナビの起点ユーザ発話を
   `get_last_chat_message(role="user")` でウィンドウ外から引き、フロントの切替は起点が表示中に
   無ければ表示を切替後の応答群で丸ごと置き換える（表示中はすべて起点より後ろなので）。
3. **`MessageList` の自動スクロール** — 着手時点で既に依存が `messages` ではなく末尾 id
   （`lastMessageId`）に変わっていたため、`scrollIntoView` のまま前方追加と衝突しない。
   コンテナ ref（`scrollRef`）は位置退避のために既にあり、フック共有の前提は満たしていた。
4. **マージ** — 1on1 は `mergeMessageWindow(prev, fresh)` 1 本（取得ウィンドウの先頭 ID が
   表示中にあればそれより手前を残す）。リロード復帰ではスナップショット（遡って読んだぶんを
   含む）を先に描き、API の直近ウィンドウをこのマージで重ねる。手前を残したときは
   `has_older` を上書きしない（サーバの答えはウィンドウ先頭に対するもので、表示の先頭とは別）。
   このため `handleSelectSession` はウィンドウ状態のリセットを**別セッションへの切替時だけ**
   行う。同じセッションの再選択（サイドバーを閉じるタップ等）でリセットすると、手前を残した
   マージ後に `has_older` が false のまま固定され、センチネルもエクスポートの全件取得も
   効かなくなる（レビューで検出）。

### そのほかの実装メモ

- **エクスポート**: 1on1 も手前が残っていれば（`hasOlderMessages`）ダイアログを開くときに
  `limit` 無しで全件を取り直す（`App.handleOpenExport`）。reasoning も全件から組み直す。
- **スナップショット**（`lib/sessionSnapshot.ts`）に `hasOlderMessages` を足した。無い古い
  退避は「1 ページ以上あれば上がある」とみなし、API 応答で正す。
- **スクロール位置の退避**には先頭メッセージの id を添える。退避時の `scrollTop` は遡って読んだ
  ぶんを含む DOM に対する値なので、サイドバーから戻ってきた直近ウィンドウだけの表示（先頭が
  違う）には当てず、最後のユーザ発話を上端に寄せる既定へ落とす。
- **前方追加の位置補正**（`useInfiniteScrollTop`）は、発火時に先頭要素の id も控え、先頭が
  変わった描画でだけ補正する。先頭が変わらない更新（送信で末尾に増えた・読み込みが早期 return
  した）では控えたまま待ち、控えた先頭がリストから消えたら捨てる。シナリオも同じフックなので
  同じ挙動になる。
- **reasoning / ログ ID のマップ**は遡って読んだぶんも `useMessageWindow` が足す
  （セッション読込時にしか復元していなかったため）。
- ページ幅 `MESSAGE_PAGE_SIZE = 60`（`api/chat.ts`。シナリオの `TURN_PAGE_SIZE` と同値）。
- 変更ファイル: `repositories/sqlite/stores/chat_store.py`・`api/chat.py`・
  `frontend/src/api/chat.ts`・`hooks/useInfiniteScrollTop.ts`（新規）・
  `hooks/useMessageWindow.ts`（新規）・`components/MessageList.tsx`・`ChatView.tsx`・
  `ScenarioChatView/index.tsx`（フックへ置換）・`App.tsx`・`lib/sessionSnapshot.ts`。
  テストは `tests/test_sqlite_chat.py`（ウィンドウ・カーソル）と `tests/test_chat_api.py`。

### 保留にした別案（棄却ではない）

- **`reasoning` の遅延取得** — 折りたたみ（`ThinkingBlock` は完了メッセージでは既定で閉）を
  開いたときに初めて取りに行く。ログビューアが既に同じ形をしている。1on1 のペイロードは
  1/5〜1/10 になり、件数ページングより効く可能性が高い。ただし**共通化には寄与しない**ため
  今回の方針では後回し。上記を入れてもまだ重ければこちらを検討する。

## 残課題

- ウィンドウ外にあらすじ境界がある場合、区切り線（`SynopsisDivider`）は表示されない。
  遡って読み込めば出る。仕様として許容する。
- 1on1 の `GET /sessions/{id}` は `limit` 省略時に全件を返す経路を残している
  （`/v1` 互換層や既存テストは触らない）。SPA からは `limit` 付きでしか呼ばない。
- `anticipation`（1.3%）はフロント未使用だが、将来 UI で使う可能性があるため今回は残す。
