/**
 * スクロールに応じてチャットヘッダーの表示/非表示を判定するフック。
 *
 * 挙動:
 *   - 内容が一画面に収まる、または最下部付近 → 表示
 *   - 上スクロール（過去の会話をさかのぼる方向）→ 非表示
 *   - 下スクロール → 表示
 *
 * 浮遊ヘッダーが過去ログの先頭に重なって読みづらくなるのを避ける目的。
 */
import { useCallback, useRef } from "react";
import type { UIEvent } from "react";

/**
 * @param onChange - ヘッダー表示状態が変わったときに呼ぶコールバック。
 * @returns スクロールコンテナに渡す onScroll ハンドラ。
 */
export function useHeaderVisibilityOnScroll(
  onChange?: (visible: boolean) => void,
) {
  /** 直前のスクロール位置（方向判定用）。 */
  const lastTop = useRef(0);

  return useCallback(
    (e: UIEvent<HTMLElement>) => {
      if (!onChange) return;
      const el = e.currentTarget;
      const top = el.scrollTop;
      // 一画面に収まる（スクロール不要）か判定する。
      const fits = el.scrollHeight <= el.clientHeight + 4;
      // 最下部付近（最新メッセージを見ている）か判定する。
      const atBottom = el.scrollHeight - top - el.clientHeight < 12;

      if (fits || atBottom) {
        onChange(true);
      } else if (top < lastTop.current - 4) {
        onChange(false); // 上スクロール → 非表示
      } else if (top > lastTop.current + 4) {
        onChange(true); // 下スクロール → 表示
      }
      lastTop.current = top;
    },
    [onChange],
  );
}
