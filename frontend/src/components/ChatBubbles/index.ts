/**
 * チャットバブル共通コンポーネント（バレル）。
 * 1on1チャット・グループチャット・シナリオで使用する表示・インタラクション部品を集約する。
 *
 * 旧 ChatBubbles.tsx（単一ファイル）を責務別に分割したパッケージ。
 * 既存の `import { ... } from "./ChatBubbles"` はこの index 経由で解決される。
 *
 * デザイン方針:
 *   キャラクターメッセージ — バブルボックスなし。アバター+名前ヘッダーの下に
 *                            インデントされたテキストを流す「ドキュメントスタイル」。
 *   ユーザーメッセージ     — 右寄せ、ニュートラルなフラット背景。
 *   ボーダー/サーフェス    — すべてニュートラルグレー。緑はキャラ名・アクセントのみ。
 */
export { charHue, bubbleClassFor } from "./colors";
export { CharacterAvatar, CharacterImageProvider } from "./avatar";
export type { CharImageResolver } from "./avatar";
export { Bubble } from "./Bubble";
export { ThinkingBlock } from "./ThinkingBlock";
export {
  CopyButton,
  DiscardButton,
  RegenerateButton,
  UserMessageActions,
} from "./buttons";
export { MessageActionBar } from "./MessageActionBar";
export {
  CharacterBubble,
  CharacterMessageRow,
  UserBubble,
  mobileBubbleExtendClass,
} from "./rows";
export { ImageGrid, ImageModal } from "./images";
export { MarkdownContent } from "./markdown";
