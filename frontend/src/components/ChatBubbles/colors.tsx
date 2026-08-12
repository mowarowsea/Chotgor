/**
 * キャラクター配色ユーティリティ。
 * キャラクター名から安定的なハッシュを取り、アバター色相・バブル配色クラスを導出する。
 *
 * バブルの色は「明示指定 > 名前ハッシュ」の優先順で決まる。明示指定はキャラクター編集
 * 画面／シナリオの NPC 編集画面で選ぶ 0〜9 のパレット番号で、`BubbleColorProvider`
 * を通じて名前から引けるようにしてある。
 */
import React, { createContext, useCallback, useContext } from "react";

/** バブル配色パレットの数（index.css の .cb0〜.cb9 と対応）。 */
export const BUBBLE_PALETTE_SIZE = 10;

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

/**
 * キャラクター名 → 明示指定のバブル配色スロット（0〜9）を解決する関数の型。
 * 明示指定が無ければ null を返し、呼び出し側は名前ハッシュにフォールバックする。
 */
export type BubbleColorResolver = (characterName: string) => number | null;

/**
 * バブル配色の明示指定リゾルバ Context。
 * 既定は「誰も明示指定していない」= 常に自動配色。
 */
const BubbleColorContext = createContext<BubbleColorResolver>(() => null);

/**
 * バブル配色の明示指定をツリーへ供給する。
 *
 * ネスト可能で、内側の resolve が null を返したら外側へ委譲する。
 * これによりシナリオ画面は「NPC の指定 → 無ければキャラクター本体の指定 → 自動」
 * という段階的な解決になる。
 */
export function BubbleColorProvider({
  resolve,
  children,
}: {
  resolve: BubbleColorResolver;
  children: React.ReactNode;
}) {
  const outer = useContext(BubbleColorContext);
  const merged = useCallback(
    (name: string) => resolve(name) ?? outer(name),
    [resolve, outer],
  );
  return (
    <BubbleColorContext.Provider value={merged}>
      {children}
    </BubbleColorContext.Provider>
  );
}

/**
 * キャラクター名から安定的なバブル配色クラス（cb0〜cb9）を返す。
 *
 * @param name キャラクター名。
 * @param explicit 明示指定のパレット番号。null/undefined・範囲外なら名前ハッシュを使う。
 */
export function bubbleClassFor(name: string, explicit?: number | null): string {
  if (
    explicit !== null &&
    explicit !== undefined &&
    Number.isInteger(explicit) &&
    explicit >= 0 &&
    explicit < BUBBLE_PALETTE_SIZE
  ) {
    return "cb" + explicit;
  }
  if (!name) return "cb5";
  return "cb" + (nameHash(name) % BUBBLE_PALETTE_SIZE);
}

/** キャラクター名からバブル配色クラスを引く。Context の明示指定を自動で見る。 */
export function useBubbleClass(name: string): string {
  const resolve = useContext(BubbleColorContext);
  return bubbleClassFor(name, resolve(name));
}
