# 記憶想起アルゴリズムの解説 (Memory Recall Algorithm)

現在の Chotgor における記憶想起の仕組みは、**「セマンティック検索（意味的類似性）」**と**「時間減衰を伴う重要度（鮮度と価値）」**を組み合わせたハイブリッド・スコアリング方式を採用しています。

## 1. 全体フロー (`MemoryManager.recall_memory`)

記憶の想起は以下の4ステップで行われます。

1.  **候補抽出 (Semantic Search)**:
    *   ChromaDB を使用して、クエリに対して意味的に近い記憶を `top_k * 2` 件抽出します。
    *   距離計算には `cosine` 類似度が使用されます。
2.  **重要度の時間減衰計算 (Time Decay)**:
    *   SQLite に保存されている各記憶の「4種類の重要度」と「最終アクセス時間」をもとに、現在のスコアを算出します。
3.  **ハイブリッド・リランク (Hybrid Reranking)**:
    *   `類似度スコア (50%)` + `時間減衰後の重要度スコア (50%)` の合算値で並び替えます。
4.  **アクセス情報の更新**:
    *   最終的に選ばれた `top_k` 件の記憶について、SQLite上の `last_accessed_at` を現在時刻に更新します（これにより「思い出した記憶」は鮮度が戻り、忘れにくくなります）。

---

## 2. 重要度と減衰のロジック ([calculate_decayed_score](file:///c:/Users/seamo/Chotgor/backend/core/memory/manager.py#16-52))

記憶の価値を以下の4つの観点で評価し、それぞれ異なる「半減期（半分に減衰する期間）」を設定しています。

| 重要度タイプ | 重み | 半減期 | 特徴 |
| :--- | :--- | :--- | :--- |
| **contextual** | 1.0 | 7日間 | 文脈的価値。短期的に重要だが、すぐに古くなる。 |
| **user** | 0.8 | 30日間 | ユーザーに関する情報。中長期的に保持される。 |
| **semantic** | 0.6 | 90日間 | 知識や概念。長期的に保持される。 |
| **identity** | 0.3 | 無限 | キャラクターのアイデンティティ。一切減衰しない。 |

**計算式:**
各重要度ごとに経過日数に応じた指数減衰 ($e^{-\lambda t}$) を計算し、重み付け合計したものが [decayed_score](file:///c:/Users/seamo/Chotgor/backend/core/memory/manager.py#16-52) となります。

---

## 3. その他の補助機能

### デイリー・ダイジェスト ([digest.py](file:///c:/Users/seamo/Chotgor/backend/core/memory/digest.py))
*   一日の終わりに（または必要に応じて）、その日の細かい記憶を Claude（内省モデル）が要約し、1つの「ダイジェスト記憶」として統合します。
*   元の細かい記憶を削除する設定（`digest_delete_originals`）も可能です。

### 忘却候補の抽出 ([get_forgotten_candidates](file:///c:/Users/seamo/Chotgor/backend/core/memory/manager.py#53-71))
*   [decayed_score](file:///c:/Users/seamo/Chotgor/backend/core/memory/manager.py#16-52) が閾値（デフォルト 0.3）を下回った記憶を、忘却（削除）の候補としてリストアップできます。
