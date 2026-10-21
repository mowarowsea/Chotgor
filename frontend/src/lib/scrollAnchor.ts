/**
 * 応答の起点（最後のユーザ発話など）をスクロールコンテナの上端へ寄せる。
 * 1on1（MessageList）とシナリオ（ScenarioChatView）で共有する。
 */

/** 浮遊ヘッダーぶんの上余白。両画面のリスト上余白（pt-16）と揃えること。 */
const FLOATING_HEADER_OFFSET = 64;

/**
 * `anchor` の頭が浮遊ヘッダーのすぐ下に来るよう `container` をスクロールする。
 *
 * 応答が短くスクロール量が足りないときはブラウザが最下端で止めるので、
 * 結果的に従来の「最下端へ寄せる」と同じ見た目になる。
 *
 * smooth は直後に別のスクロール（最下端追従など）が走ると打ち切られる点に注意。
 */
export function scrollAnchorToTop(
  container: HTMLElement,
  anchor: Element,
  behavior: ScrollBehavior,
): void {
  const top =
    container.scrollTop +
    anchor.getBoundingClientRect().top -
    container.getBoundingClientRect().top -
    FLOATING_HEADER_OFFSET;
  container.scrollTo({ top, behavior });
}
