# Chotgor Documents

ドキュメントは用途別に3つへ分けています。

## current-spec

現仕様・現実装の参照用。

- [ARCHITECTURE.md](current-spec/ARCHITECTURE.md): システム地図、主要フロー、関連文書一覧
- [memory_recall_algorithm.md](current-spec/memory_recall_algorithm.md): 記憶想起アルゴリズム
- [character_resident_rules.md](current-spec/character_resident_rules.md): キャラクター向け仕様
- [fable_view.md](fable_view.md): キャラクター当事者視点の観測記録（設計判断の根拠）

## explain

説明用・案内用。

- [README.md](explain/README.md): プロジェクト紹介
- [DEAR_GHOST.md](explain/DEAR_GHOST.md): キャラクター向けシステムガイド
- [frontend_start.txt](explain/frontend_start.txt): フロント起動メモ

## planned

これから実装・検討する設計案、または計画履歴。

- [aliveness_plan.md](planned/aliveness_plan.md): めぐり / Aliveness 計画
- [schedule_plan.md](planned/schedule_plan.md): 生活カレンダー設計
- [usual_days_plan.md](planned/usual_days_plan.md): うつつ / Usual Days 計画
- [forecast_panel_plan.md](planned/forecast_panel_plan.md): 予報パネル設計

## old

陳腐化してアーカイブした文書。現仕様の参照には使わない（残すのは経緯確認のため）。

- [backend_data_design.md](old/backend_data_design.md): 旧データ設計まとめ。撤去済みテーブル（SessionDrift 等）を現役記載し、めぐり以降の新テーブルを欠く。現行のテーブル定義は `backend/repositories/sqlite/models.py` と `migrations.py` が正。

## Root Files

- [AGENTS.md](../AGENTS.md): Codex / agent 向けのプロジェクト原則。ツールが読むためルートに残す。
- [CLAUDE.md](../CLAUDE.md): Claude Code 向けの開発ガイド。ツールが読むためルートに残す。
