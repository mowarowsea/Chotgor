/**
 * 履歴の上端に近づいたら過去を 1 ページぶん自動で取りに行く「無限スクロール」フック。
 *
 * 1on1（MessageList）とシナリオ（ScenarioChatView）で共有する。取得層はそれぞれ別
 * （カーソルや API が違う）だが、スクロール機構は同じなのでここに一本化する。
 *
 * 担うこと:
 *   - リスト先頭のセンチネル要素を `IntersectionObserver` で監視し、可視域に入ったら
 *     `onLoadOlder` を呼ぶ（root はスクロールコンテナ。上方向に先読みマージン）。
 *   - 発火時に「下端からの距離（scrollHeight - scrollTop）」を控え、前方に要素が
 *     積まれて DOM が伸びた直後（`useLayoutEffect`）に同じ距離へ戻して、見ている位置が
 *     ずれないよう補正する。
 *
 * 担わないこと:
 *   - 最下端への自動追従。呼び出し側が「末尾の id が変わったときだけ」追従させる
 *     （前方に足しても末尾は変わらないので、遡り読み込みでは発火しない）。
 *
 * `IntersectionObserver` は交差状態が「変化した」ときしか呼ばれない。1 ページ足しても
 * センチネルがまだ可視域に残っている（1 ページでは埋まらなかった）場合は再発火しないので、
 * `loading` を依存に入れて読み込み完了のたびに購読し直し、その場合も続けて伸ばす。
 */
import { useEffect, useLayoutEffect, useRef } from "react";
import type { RefObject } from "react";

interface Options {
  /** 表示中ウィンドウより過去がまだ残っているか。false ならセンチネルを監視しない。 */
  hasOlder: boolean;
  /** 遡り読み込みの実行中か。true の間は多重発火させない。 */
  loading: boolean;
  /** 表示ウィンドウを 1 ページぶん過去へ伸ばす。 */
  onLoadOlder?: () => void;
  /**
   * 前方追加の補正を走らせる依存値。表示リスト（turns / messages）そのものを渡す。
   * この値が変わった描画の直後に、控えておいた下端からの距離へ戻す。
   */
  items: unknown;
  /** 上端に届く前から先読みするマージン（px）。 */
  prefetchMargin?: number;
}

/**
 * @param scrollRef - スクロールコンテナの ref。
 * @returns リスト先頭に置くセンチネル要素の ref。`hasOlder` のときだけ描画すればよい。
 */
export function useInfiniteScrollTop(
  scrollRef: RefObject<HTMLElement>,
  { hasOlder, loading, onLoadOlder, items, prefetchMargin = 600 }: Options,
): RefObject<HTMLDivElement> {
  const sentinelRef = useRef<HTMLDivElement>(null);
  /** 発火時点の「下端からの距離」。null なら補正待ちではない。 */
  const anchorRef = useRef<number | null>(null);
  /** 親から渡る関数は毎レンダー変わりうるので、購読を張り直さずに済むよう ref 経由で呼ぶ。 */
  const onLoadOlderRef = useRef(onLoadOlder);
  onLoadOlderRef.current = onLoadOlder;

  // 前方に積まれたぶんのずれを打ち消す。描画前に補正したいので useLayoutEffect。
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const anchor = anchorRef.current;
    if (el && anchor !== null) {
      el.scrollTop = el.scrollHeight - anchor;
      anchorRef.current = null;
    }
  }, [items, scrollRef]);

  useEffect(() => {
    if (!hasOlder || loading) return;
    const root = scrollRef.current;
    const target = sentinelRef.current;
    if (!root || !target) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries[0]?.isIntersecting) return;
        anchorRef.current = root.scrollHeight - root.scrollTop;
        onLoadOlderRef.current?.();
      },
      { root, rootMargin: `${prefetchMargin}px 0px 0px 0px` },
    );
    io.observe(target);
    return () => io.disconnect();
  }, [hasOlder, loading, prefetchMargin, scrollRef]);

  return sentinelRef;
}
