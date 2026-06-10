/**
 * バックエンドAPI呼び出し層（バレル）。
 *
 * 旧 api.ts（単一ファイル）をドメイン別に分割したパッケージ。
 * 既存の `import { ... } from "../api"` はこの index 経由で解決される。
 *
 * - chat.ts      — 1on1チャット・グループチャット・モデルIDユーティリティ
 * - scenario.ts  — シナリオチャット（テンプレ・セッション・SSE）
 * - logs.ts      — デバッグログ閲覧
 * - translate.ts — テキスト翻訳
 * - sse.ts       — SSE 共通パーサ
 */
export * from "./chat";
export * from "./scenario";
export * from "./logs";
export * from "./translate";
