/**
 * 「いまどのバブルの操作ボタンを出しているか」だけを持つ最小ストア。
 *
 * 全バブルにボタンを置いて CSS で透明にする方式だと、発話数ぶんの button/svg が
 * DOM に積み上がる（長いシナリオでは数百バブル）。露出中の 1 つだけを実際に
 * 描画するため、露出先をグローバルに 1 つだけ持つ。副次的に「画面内で鉛筆は
 * 常に 1 つ」も満たす。
 *
 * React state ではなく外部ストアにしているのは、露出先が変わったときに
 * 一覧コンポーネント全体を再レンダリングさせないため。購読側のスナップショットは
 * boolean なので、状態が実際に変わったバブル（高々 2 つ）だけが再描画される。
 */
import { useSyncExternalStore } from "react";

let revealedId: string | null = null;
const listeners = new Set<() => void>();

function subscribe(onChange: () => void) {
  listeners.add(onChange);
  return () => {
    listeners.delete(onChange);
  };
}

/** 露出させるバブルを差し替える（null で閉じる）。前に開いていたものは自動的に閉じる。 */
export function revealBubble(id: string | null) {
  if (revealedId === id) return;
  revealedId = id;
  for (const fn of listeners) fn();
}

/** 自分が露出中かを購読する。 */
export function useBubbleRevealed(id: string): boolean {
  return useSyncExternalStore(
    subscribe,
    () => revealedId === id,
    () => false,
  );
}
