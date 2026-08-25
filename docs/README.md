# Chotgor Documents

ドキュメントは用途別に4つへ分けています。

- **迷ったらまず** [current-spec/ARCHITECTURE.md](current-spec/ARCHITECTURE.md)。システム地図であり、
  各機能の詳細設計書（`planned/`）への入口も兼ねます。

## current-spec

現仕様・現実装の参照用。実装が変わったら同じコミットで直す。

- [ARCHITECTURE.md](current-spec/ARCHITECTURE.md): システム地図、主要フロー、関連文書一覧
- [memory_recall_algorithm.md](current-spec/memory_recall_algorithm.md): 記憶想起アルゴリズムと注入量の規定
- [character_resident_rules.md](current-spec/character_resident_rules.md): キャラクター向け仕様

## explain

説明用・案内用。

- [DEAR_GHOST.md](explain/DEAR_GHOST.md): キャラクター向けシステムガイド（世界の歩き方）
- [frontend_start.txt](explain/frontend_start.txt): フロント起動メモ
- [README.md](explain/README.md): プロジェクト紹介（※歴史的紹介文。現仕様は ARCHITECTURE.md が正）
- [fable_view.md](fable_view.md): エンジン（Fable 5）当事者視点の観測記録（設計判断の根拠・2026-07-11）

## planned

機能ごとの詳細設計書。**現時点ではすべて実装済み**（`prompt_cache_plan.md` のみ一部残あり）で、
ARCHITECTURE.md が骨子、こちらが「なぜそう作ったか・何を棄却したか」を持つ。
名前が `planned/` なのは「設計→実装の記録」を溜める場所という経緯によるもの。

| 文書 | 内容 | 実装 |
|---|---|---|
| [usual_days_plan.md](planned/usual_days_plan.md) | うつつ（Usual Days）— ユーザ不在時の無人生活モード | 2026-06-14 |
| [aliveness_plan.md](planned/aliveness_plan.md) | めぐり（巡り / Aliveness）— タイムライン正本・可視性・計器・動機経済 | 2026-07-07 |
| [prompt_cache_plan.md](planned/prompt_cache_plan.md) | プロンプトキャッシュ有効化（システム／ターン注釈の二層分離） | 2026-07-07（残: C案・第2段計測） |
| [schedule_plan.md](planned/schedule_plan.md) | 生活カレンダー（Living Schedule）— 週次バッチ・占有圧・配達値・③突発 | 2026-07-09 |
| [forecast_panel_plan.md](planned/forecast_panel_plan.md) | 予報パネル — 決定ログ・heartbeat・無風外挿・配達シミュレータ | 2026-07-10 |
| [switch_angle_removal_plan.md](planned/switch_angle_removal_plan.md) | switch_angle 機能の全面撤去 | 2026-07-19 |
| [block_naming_cleanup_plan.md](planned/block_naming_cleanup_plan.md) | システムプロンプトのブロック命名整理（`provider_extra`→`session_frame`・Block N 廃止） | 2026-07-19 |
| [ambience_plan.md](planned/ambience_plan.md) | なりゆき（ambience）— farewell の再解釈・judge 実名化・対面背景の切替 | 2026-07-19 |
| [speak_later_plan.md](planned/speak_later_plan.md) | 発話予約（speak_later）— キャラ発の時限発話 | 2026-07-20 |
| [scenario_turn_variants_plan.md](planned/scenario_turn_variants_plan.md) | シナリオログの枝分かれ（レスポンスガチャ）と手動書き換え | 2026-07-28 |
| [scenario_history_perf_plan.md](planned/scenario_history_perf_plan.md) | シナリオ履歴の転送・描画コスト削減（ウィンドウ＋無限スクロール） | 2026-07-31 |
| [sse_disconnect_resilience_plan.md](planned/sse_disconnect_resilience_plan.md) | SSE 切断耐性 — 応答を接続の生死から切り離す | 2026-08-05 |
| [audio_attachment_plan.md](planned/audio_attachment_plan.md) | 音声添付（曲を聴かせる）— 添付の汎用化・寿命の統一 | 2026-08-20 |
| [wm_repeat_awareness_plan.md](planned/wm_repeat_awareness_plan.md) | WM 繰り返し話題への気づき誘導（Open×Close 類似検出＋内省誘導） | 2026-08-21 |

## old

陳腐化してアーカイブした文書。現仕様の参照には使わない（残すのは経緯確認のため）。

- [backend_data_design.md](old/backend_data_design.md): 旧データ設計まとめ。撤去済みテーブル（SessionDrift 等）を現役記載し、めぐり以降の新テーブルを欠く。現行のテーブル定義は `backend/repositories/sqlite/models.py` と `migrations.py` が正。

## Root Files

- [AGENTS.md](../AGENTS.md): Codex / agent 向けのプロジェクト原則。ツールが読むためルートに残す。
- [CLAUDE.md](../CLAUDE.md): Claude Code 向けの開発ガイド。ツールが読むためルートに残す。
