/** シナリオチャットビュー共通の文字列ヘルパー。 */

/** 文字列の末尾空白・改行を取り除く（表示時のノイズ除去）。 */
export function trimEnd(s: string): string {
  return s.replace(/\s+$/u, "");
}

/** アバター用のプレースホルダー文字（名前の頭文字）。 */
export function avatarInitial(name: string): string {
  return name?.[0] ?? "?";
}
