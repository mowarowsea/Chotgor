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
import type React from "react";
import { useId, useSyncExternalStore } from "react";

let revealedId: string | null = null;
const listeners = new Set<() => void>();

function subscribe(onChange: () => void) {
  listeners.add(onChange);
  return () => {
    listeners.delete(onChange);
  };
}

/** 露出させるバブルを差し替える（null で閉じる）。前に開いていたものは自動的に閉じる。 */
function revealBubble(id: string | null) {
  if (revealedId === id) return;
  revealedId = id;
  for (const fn of listeners) fn();
}

/** 自分が露出中かを購読する。 */
function useBubbleRevealed(id: string): boolean {
  return useSyncExternalStore(
    subscribe,
    () => revealedId === id,
    () => false,
  );
}

/**
 * 1 バブルぶんの露出制御。1on1 / グループ / シナリオの全モードで共有する。
 *
 * 露出のきっかけはマウスが行ホバー、タッチがバブルのタップ。タッチ環境でも
 * mouse 相当のイベントが合成されるため `pointerType` で振り分ける（振り分けないと、
 * タップで開いた直後に click 相当が来て即座に閉じてしまう）。
 *
 * `pinned` を渡したバブルはホバーを待たずに常時露出する。破棄・削除の直後は
 * 再レンダとレイアウト再計算でホバーの反映が一拍遅れ、「押したいのにボタンが出ない」
 * 時間が生まれるため、いま操作できる末尾バブルだけはきっかけ無しで出しておく。
 * 常時露出でもハンドラは配る — 自分の上へマウスが来たら露出先を自分へ移し、
 * 別の場所で開きっぱなしのボタンを閉じるため（画面内で 1 つだけ、を保つ）。
 *
 * @param enabled false のときハンドラを配らない（送信中・終了セッション等）。
 * @param pinned true のときホバー非依存で常時露出する（操作可能な末尾バブル）。
 * @returns `revealed`（描画すべきか）、行へ広げる `rowProps`、バブルへ広げる `bubbleProps`。
 */
export function useRevealControls(enabled: boolean, pinned = false): {
  revealed: boolean;
  rowProps: {
    onPointerEnter?: React.PointerEventHandler;
    onPointerLeave?: React.PointerEventHandler;
  };
  bubbleProps: { onPointerUp?: React.PointerEventHandler };
} {
  const id = useId();
  const hovered = useBubbleRevealed(id);
  // enabled=false は送信中・編集中。この間は pinned でも出さない（押せない操作を見せない）。
  if (!enabled) return { revealed: false, rowProps: {}, bubbleProps: {} };
  return {
    revealed: hovered || pinned,
    rowProps: {
      onPointerEnter: (e) => {
        if (e.pointerType === "mouse") revealBubble(id);
      },
      onPointerLeave: (e) => {
        if (e.pointerType === "mouse") revealBubble(null);
      },
    },
    bubbleProps: {
      onPointerUp: (e) => {
        // タップのトグルはホバー相当の露出だけを見る（pinned は常時出しっぱなしなので対象外）。
        if (e.pointerType !== "mouse") revealBubble(hovered ? null : id);
      },
    },
  };
}
