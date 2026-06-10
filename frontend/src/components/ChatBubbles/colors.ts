/**
 * キャラクター配色ユーティリティ。
 * キャラクター名から安定的なハッシュを取り、アバター色相・バブル配色クラスを導出する。
 */

/** キャラクター名から安定的なハッシュ値（符号なし32bit）を生成する。 */
function nameHash(name: string): number {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return h;
}

/** キャラクター名から安定的な色相（0–359）を返す。アバター配色に使う。 */
export function charHue(name: string): number {
  if (!name) return 200;
  return nameHash(name) % 360;
}

/** キャラクター名から安定的なバブル配色クラス（cb0〜cb9）を返す。 */
export function bubbleClassFor(name: string): string {
  if (!name) return "cb5";
  return "cb" + (nameHash(name) % 10);
}
