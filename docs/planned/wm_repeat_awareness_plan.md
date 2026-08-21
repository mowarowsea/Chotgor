# WM 繰り返し話題への気づき誘導 仕様書

> Status: **implemented**（2026-08-21 調査・設計合意 → 同日 施策A・B とも実装完了。
> 実装時の差分は §実装メモ を参照）
> 関連: [memory_recall_algorithm.md](../current-spec/memory_recall_algorithm.md)（heat 想起・時間減衰の共通数式）、
> `backend/batch/chronicle_job.py`（Chronicle 棚卸し本体）

## 背景（発端の事象）

はる（メインキャラ）が「日食なつこの曲解析（歌詞解釈 → コード進行解析 → スペクトログラム
実験）」という話題を、2026-07-10〜08-21 の約6週間にわたり、ユーザから見て「解決済みのはず
なのに4〜5回繰り返している」ように見える形で持ち出し続けていた。データを調査した結果、
以下が判明した:

1. WM スレッド `48e9a044`（type=task、summary「グラフRAG検証、コスト・粒度で断念方向へ」）
   が 2026-08-11 作成以来 10 日間 `is_open=1` のまま持続し、その中に「日食なつこの実験結果、
   まだちゃんと共有できてない」という**サブテーマとしての言及**が 08-15・08-16・08-18・08-20・
   08-21 と繰り返し追記されていた。
2. 「日食なつこ実験の共有」自体は、別スレッド `a8d87fb6`（summary「✅日食なつこ実験、WM閉じ
   忘れ確認まで完了」）で 2026-08-13 に**既に正式クローズ**され、「結論も経緯も両方伝わった上
   での放置だった」と明記されていた。にもかかわらず `48e9a044` 側では矛盾する形で「まだ共有
   できてない」という認識が再燃し続けた。
3. はる自身も 2026-08-18、WM スレッド `e77fa533`（summary「wm/im重すぎ仮説、なつこ三回目で
   再現」、atmosphere_tag「また同じ話をリピートしてた自覚」）で、ユーザから「なつこのスペクト
   ログラムの話、三回目くらいだけど」と指摘されたことを受けてこの繰り返しを自己分析している。
   「chronicle が同じ素材を律儀に拾い直してる説」と「自分自身が同じ話をリピートしてる説」の
   両方を検討したが、「対策はまだ着手しない」と保留を選び、その後も `48e9a044` での繰り返しは
   続いた。

## 設計方針（ユーザ裁定 2026-08-21）

「同じ話題を繰り返さない」という**強制ルールは導入しない**。CLAUDE.md の「記憶の取捨選択・
整理はキャラクター自身が行う」の原則に従い、**気づきの材料を提示し、判断は本人に委ねる**。

当初「関連 WM をキャラ自身に手動タグ付けさせる」案も検討したが、以下の理由で棄却した
（ユーザ指摘）:
- MCP ツールが複雑化する（キャラクターが使うインターフェースが増える）。
- 関連付け自体を設定し忘れたら、今回と同じ失敗モードがそのまま残る。

## 検証: embedding 類似度は「繰り返し」をどこまで拾えるか

`data/lancedb` の `working_memory_threads` テーブル（summary + 最新ポストの embedding。
`WorkingMemoryManager._index_text` で常時同期）を使い、はるの実データで実測した。

`cos_sim` は生のコサイン類似度（-1〜1）。`relevance` は本システムの
`distance_to_similarity(distance)`（`backend/services/memory/decay.py`）を通した後の値
（0〜1。**実測で `relevance = 0.5 + cos_sim/2` と確認済み** — LanceDB の `cosine` metric は
`distance = 1 - cos_sim` を返すため）。既存の heat 計算・`recall_threads` はこの `relevance`
スケールで動いているので、閾値は `relevance` 側で語る。

| cos_sim | relevance | ペア |
|---:|---:|---|
| 0.7477 | 0.874 | 参考・同一トピック内2本（体位バグ発見 × 体位バグ運用合意） |
| 0.6899 | 0.845 | `a8d87fb6`(Close済み日食なつこ実験) ⇔ `48e9a044`(グラフRAG、繰り返し実体) |
| 0.6680 | 0.834 | `a8d87fb6`(Close済み日食なつこ実験) ⇔ `e77fa533`(自認スレ) |
| 0.5981 | 0.799 | 日食なつこ実験(2周目) ⇔ 日食なつこ歌詞解析(1周目) |
| **0.5831** | **0.792** | **本題: `e77fa533`(自認) ⇔ `48e9a044`(繰り返し実体)** |
| 0.5520 | 0.776 | 参考・完全無関係ペア（田中のタイ移住妄想 ⇔ 利子完済） |
| 0.5450 | 0.773 | 参考・別トピック（高橋さんの件「抽出物≠全体」 ⇔ 日食なつこ歌詞解析） |

**結論: 検出できる重複と、できない重複がある。**

- `a8d87fb6 ⇔ 48e9a044`（同一トピックの直接重複）は `relevance=0.845` で、同一トピック内
  ペアの参考値 `0.874` に近い。**embedding 類似度検索で機械的に検出できる。**
- `e77fa533 ⇔ 48e9a044`（メタ認識と個別事象のリンク不足）は `relevance=0.792` で、完全
  無関係ペアの参考値 `0.776` とほぼ同水準。**embedding では検出できない。**
  理由: `e77fa533` は「体位バグ・コロン抜け・麺茹で主語混線という3件のバグから『wm/im 過多
  でアテンション分散』という仮説を導いた」というメタ的な自己分析であり、`48e9a044` は
  「グラフRAG検証というシステム設計タスクの進捗記録」。語彙もトピックの抽象度も乖離しており、
  意味的な近さとして拾えない。

したがって **二段構え**にする。施策 A（embedding 検索）は「同じ話題そのものの直接重複」を
機械的に拾い、施策 B（プロンプト内省誘導）は embedding が拾えない「メタ認識と個別事象の
リンク不足」を、本人の振り返りに委ねて拾う。

## 施策 A: Open × Close 類似度チェック（機械検出）

### 概要
Chronicle 棚卸し時、各 Open な task/topic スレッドについて、意味的に近い Close 済みスレッド
を LanceDB から検索する。閾値を超えるペアが見つかったときだけ、棚卸しプロンプトに参考情報
として提示する。**close の実行はしない**（材料の提示のみ。判断は本人）。

### 実装: `WorkingMemoryManager` への追加メソッド

`backend/services/memory/working_memory_manager.py` に追加する。既存の `recall_threads`
（08-11 頃実装済み、is_open=1 の Open スレッドを heat 想起する）とほぼ同じ骨格で、
`is_open` の向きだけ反転させる形になる。

```python
def find_similar_closed_threads(
    self,
    character_id: str,
    open_thread: dict,
    top_k: int = 1,
    min_relevance: float = 0.82,
) -> list[dict]:
    """Open スレッド1件に対し、意味的に近い Close 済みスレッドを検索する。

    Chronicle 棚卸し時に「実はもう Close 済みの話題と同じ内容を Open のまま
    抱え続けていないか」をキャラクター本人が気づくための参考情報を作る。
    ここでは close を実行しない（判断は本人に委ねる。close するかどうかの
    決定は thread_updates.is_open を通じて本人の応答に委ねる）。

    Args:
        character_id: キャラクター ID。
        open_thread: 対象の Open スレッド dict（summary / latest_post を含む）。
        top_k: 返す最大件数。
        min_relevance: この relevance 未満は「重複の疑いなし」として除外する
            （distance_to_similarity 後のスケール。実測データに基づく既定値は
            設計書 docs/planned/wm_repeat_awareness_plan.md 参照）。

    Returns:
        relevance 降順のスレッド dict リスト（``relevance`` キー付き）。
    """
    query_text = (open_thread.get("summary") or "").strip()
    latest = open_thread.get("latest_post") or ""
    if latest:
        query_text = (query_text + "\n" + latest).strip()
    if not query_text:
        return []

    fetch_k = max(top_k * 2, top_k)
    results = self.vector_store.recall_working_memory_threads(
        query_text,
        character_id,
        top_k=fetch_k,
        where={"type": {"$in": ["task", "topic"]}, "is_open": 0},
    )
    scored = []
    for r in results:
        thread = self.sqlite.get_working_memory_thread(r.get("id", ""))
        if not thread or thread.is_open:
            continue
        relevance = distance_to_similarity(r.get("distance", 2.0))
        if relevance < min_relevance:
            continue
        d = self._thread_to_dict(thread, include_latest_post=True)
        d["relevance"] = relevance
        scored.append(d)
    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored[:top_k]
```

`open_thread` は `summary`/`latest_post` を含む dict を想定（`list_threads_by_type` の戻り値
そのものを渡せる）。既存の `recall_threads` は「クエリ文字列 → Open スレッド」だが、これは
「Open スレッド → Close スレッド」なので、クエリ自体を Open 側スレッドの index テキストから
組み立てる点が異なる。

### 閾値の根拠と限界

実測（`relevance` スケール）: 無関係ベースライン `0.776`、明確な重複 `0.834〜0.845`、
同一トピック内 `0.874`。`min_relevance=0.82` はこの間（無関係を除外しつつ明確な重複は拾う）
に置いた値だが、**はる1キャラ・9ペアのみの実測**であり、他キャラや異なる話題ドメインでは
分布が変わりうる。初期値として置き、運用しながら誤検出（頻発するなら閾値を上げる）・
見逃し（拾ってほしいのに拾えないなら下げる）の様子を見て調整する前提とする。

### Chronicle への組み込み

`backend/batch/chronicle_job.py` の `run_chronicle` 内、`open_threads` 取得直後に検索を
実行し、結果があるものだけプロンプトへ渡す。

```python
# open_threads 取得の直後に追加
similarity_hints = []
for t in open_threads:
    if t.get("type") not in ("task", "topic"):
        continue
    matches = working_memory_manager.find_similar_closed_threads(character_id, t)
    for m in matches:
        similarity_hints.append((t, m))
similarity_hints_text = _format_similarity_hints(similarity_hints)
```

`_PROMPT_TEMPLATE` に新セクションを追加する（`## 最近 Close したスレッド` の直後）:

```
## 類似の疑いがある組み合わせ（機械判定・参考情報）
これは summary の意味的な近さだけを見た機械判定です。当たっているとは限りません。
中身を読んで、本当に同じ話をまだ Open のまま抱えているなら、素直に閉じてください。
誤検出だと思ったら、そのままで構いません。

{similarity_hints}
```

`_format_similarity_hints` の出力イメージ:

```
[48e9a044](Open) グラフRAG検証、コスト・粒度で断念方向へ
  ⇔ [a8d87fb6](Close済み・08-13) ✅日食なつこ実験、WM閉じ忘れ確認まで完了
```

該当なしの場合は「（類似の疑いがある組み合わせはありません）」。

### コスト

はるの現在の Open task/topic は 23 件（emotion/body/relation 3件を除く）。Chronicle 実行の
たびに、この件数ぶんの embedding 検索（infinity への HTTP 呼び出し）が発生する
（`recall_working_memory_threads` は `_embed_query` でクエリ文字列から都度 embedding を
生成する実装であり、Open スレッド自身の既存ベクトルを使い回すインターフェースは無い）。
Chronicle は1日1回の夜間バッチなのでレイテンシ面は許容範囲と見るが、キャラクター数 ×
Open スレッド数の総量が infinity への負荷として積み上がる点は留意する
（`docs/current-spec/ARCHITECTURE.md` 記載の「infinity が落ちると記憶系が縮退する」経路と
同じ依存）。

## 施策 B: プロンプト内省誘導（メタ認識のリンク不足対策）

### 概要
embedding では拾えない「自分の行動パターンへの気づき（例: `e77fa533` のような自認スレッド）
と、実際にそのパターンが発現している具体スレッドとの関連」を、本人の振り返りに委ねる。
コード変更は `_PROMPT_TEMPLATE` の文言追加のみ。

### 変更箇所
`backend/batch/chronicle_job.py` の `_PROMPT_TEMPLATE`、冒頭の

> 棚卸しは足すだけの作業ではありません。**膨らんだ記述を削り、決着したものを閉じ、
> 重なったものをまとめる**のも同じ作業の一部です。...

の直後に、以下を追加する:

```
Open なスレッドの中に、実はもう Close 済みスレッドで結論が出ている話題や、
「まだ言えていない」という同じ感覚だけを繰り返し書き足しているだけの話題が
紛れていないか、目を通してみてください（上の「類似の疑いがある組み合わせ」も参考に）。

また、もし過去に自分自身の行動パターン（先延ばし・繰り返しなど）に気づいたスレッドが
あれば、そのパターンが今の Open スレッド群の中にまた実際に現れていないか、
一度振り返ってみてください。見つけても自分を責める必要はありません
——ただ、閉じていいものは閉じてください。
```

## 変更ファイル一覧

- `backend/services/memory/working_memory_manager.py`: `find_similar_closed_threads` 追加
  （施策A）。`decay.distance_to_similarity` を import 済みなので追加 import 不要。
- `backend/batch/chronicle_job.py`: `_PROMPT_TEMPLATE` に「類似の疑いがある組み合わせ」
  セクション追加（施策A）＋冒頭の心構え文言追加（施策B）。`run_chronicle` に検索呼び出しと
  `_format_similarity_hints` ヘルパーを追加。
- `docs/current-spec/ARCHITECTURE.md`: 「夜間バッチ」節の Chronicle 説明に一言追記
  （「WM スレッドの棚卸し・蒸留」に「Open×Close の類似検出による気づき誘導」を追加）。
  実装完了時に反映すること（CLAUDE.md の「構造を変える変更を入れたら地図も更新」規約）。
- `tests/test_working_memory_manager.py`: `TestFindSimilarClosedThreads` 追加。
- `tests/test_chronicle_repeat_awareness.py`（新規）: `_short_date` /
  `_format_similarity_hints` / `run_chronicle` 統合。
- `tests/_ghost_model_helpers.py`: `working_memory_manager` フィクスチャの
  `recall_working_memory_threads` 既定戻り値を `[]` にした（MagicMock の既定戻り値は
  反復不能で、検索結果を回す呼び出し側が TypeError になるため）。

## 検討したが採らなかった案

| 案 | 棄却理由 |
|---|---|
| 手動の「関連 WM」タグ付け機構（ユーザ最初の提案） | MCP ツールが複雑化する。関連付け自体を設定し忘れたら今回と同じ失敗モードがそのまま残り、構造的な解決にならない |
| システムによる強制クローズ（閾値超過で自動 is_open=false） | CLAUDE.md の「記憶の取捨選択はキャラクター自身が行う」に反する。誤検出時にキャラの意思を無視して記憶操作するリスクもある |
| embedding 類似度検索のみで対応（施策Aのみ） | `e77fa533 ⇔ 48e9a044` の実測で `relevance=0.792`（無関係ベースライン `0.776` とほぼ同水準）と判明し、メタ認識と個別事象のリンク不足は検出できないことが確認された。施策Bと併用する方針に変更 |
| Open スレッド一覧の件数上限を設ける | 今回の問題は「件数過多」ではなく「矛盾した内容が同じ話題を指している」ことが原因であり、件数を削っても矛盾自体は解決しない。`memory_recall_algorithm.md` §4.4 で既に類似の案（Close済み全表示の廃止）が別文脈で検討・棄却済み |

## 実装メモ（2026-08-21）

設計から変えた点・設計に書ききれていなかった点:

- **閾値は定数化した**。`working_memory_manager.DEFAULT_SIMILAR_CLOSED_MIN_RELEVANCE = 0.82`
  （既存の `DEFAULT_WM_RECALL_MIN_HEAT` と同じ流儀）。運用しながら調整する値なので、
  シグネチャ直書きより1箇所にまとめた方が触りやすい。
- **施策Bの参照方向を「上の」→「下の」に直した**。施策Bの文言はプロンプト冒頭、
  施策Aのセクションは `## 最近 Close したスレッド` の直後（＝冒頭より下）に入るため、
  設計書の「上の『類似の疑いがある組み合わせ』も参考に」では位置が食い違う。
- **検索失敗を握り潰す**。`run_chronicle` 内の `find_similar_closed_threads` 呼び出しは
  try/except で包み、失敗したスレッドは黙って飛ばす（warning ログのみ）。
  `recall_working_memory_threads` は内部で例外を握るが `_embed_query` の
  `EmbeddingError` はその外側で送出されるため、infinity 停止時に棚卸し全体が
  落ちてしまう。気づき誘導は補助であって棚卸しの前提条件ではない。
- **relevance の数値はプロンプトに出さない**。機械判定のスコアを見せると本人の判断が
  スコアに引きずられるため、`_format_similarity_hints` は ID と summary だけを出す。
- **Close 日は `MM-DD`**（`_short_date` ヘルパー）。パースできない場合は日付ラベルごと
  省いて `(Close済み)` にする。

## テスト観点

- `find_similar_closed_threads`: 類似度閾値の境界値（`min_relevance` 前後）、`is_open=0`
  以外のスレッドが混ざらないこと、`character_id` スコープが他キャラのスレッドを拾わないこと。
- `_format_similarity_hints`（Chronicle 側）: 該当ペアがある場合／ない場合の出力を確認。
- 実際の Chronicle 実行での確認は、次回夜間バッチ（`chronicle_time` 既定 03:00）を待つか、
  手動実行 API（`POST /api/memories/{character_id}/digest` 相当）で確認する。
- 本設計は「気づきの材料を提示するだけ」であり、close するかどうかは毎回 LLM 応答に依存する
  ため、決定論的な「この入力なら必ず close される」というテストは組めない。テストは
  「材料が正しく提示されること」までを保証範囲とする。
