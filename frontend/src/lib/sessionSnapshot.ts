/**
 * 直近に開いていた 1on1 セッションの画面状態を sessionStorage へ退避し、
 * リロード直後に即描画するためのモジュール（stale-while-revalidate のキャッシュ層）。
 *
 * モバイルブラウザはタブを離れている間にタブを破棄することがあり、戻ると必ずリロードになる。
 * その復帰で API 応答を待つと、数百 ms〜数秒のあいだ空の画面が出てテンポを損なう。
 * ここに直前の描画内容を置いておき、まずそれを描いてから裏で最新を取り直す。
 *
 * 設計上の注意:
 * - sessionStorage を使うのはタブ単位で完結させるため（別タブの操作と混ざらない）。
 *   タブ破棄からの復帰は同一タブの復元として扱われるため、内容は残っている。
 * - 保存対象は 1on1 のみ。シナリオは履歴ウィンドウ方式で描画側の前提が違うため対象外。
 * - 保持するのは最後に開いた 1 セッションだけ。復帰時に要るのはそれだけで、
 *   全セッション分を抱えると容量上限（数 MB）に当たりやすくなる。
 * - 書き込み失敗（容量超過・プライベートモード）は握りつぶす。スナップショットが無くても
 *   通常の取得経路で復帰できる、あくまで体感短縮のためのキャッシュ。
 */
import type { ChatMessage } from "../api";

/** セッション内容の退避キー。 */
const SNAPSHOT_KEY = "chotgor:session-snapshot";
/** スクロール位置の退避キー。 */
const SCROLL_KEY = "chotgor:scroll-pos";

/** リロード復帰時に即描画するためのセッション画面状態。 */
export interface SessionSnapshot {
  sessionId: string;
  /** 選択中モデルID（"{char_name}@{preset_name}"）。空なら復元しない。 */
  modelId: string;
  messages: ChatMessage[];
  reasoningMap: Record<string, string>;
  msgLogIds: Record<string, string>;
}

/** 退避したスクロール位置。 */
export interface ScrollPos {
  scrollTop: number;
  /** 最下部付近を見ていたか。true なら復元せず最新へ寄せる。 */
  atBottom: boolean;
}

/**
 * 指定セッションのスナップショットを取り出す。
 * 別セッションのものしか無い場合は null（誤ったセッションの内容を描かない）。
 */
export function loadSessionSnapshot(sessionId: string): SessionSnapshot | null {
  try {
    const raw = sessionStorage.getItem(SNAPSHOT_KEY);
    if (!raw) return null;
    const snap = JSON.parse(raw) as SessionSnapshot;
    return snap.sessionId === sessionId ? snap : null;
  } catch {
    return null;
  }
}

/** スナップショットを退避する。失敗は無視する（キャッシュなので必須ではない）。 */
export function saveSessionSnapshot(snap: SessionSnapshot): void {
  try {
    sessionStorage.setItem(SNAPSHOT_KEY, JSON.stringify(snap));
  } catch {
    // 容量超過・プライベートモード。復帰が少し遅くなるだけなので黙って諦める。
  }
}

/** スナップショットを破棄する（対象セッションが消えていた場合など）。 */
export function clearSessionSnapshot(): void {
  try {
    sessionStorage.removeItem(SNAPSHOT_KEY);
  } catch {
    // 同上。
  }
}

/** 指定セッションのスクロール位置を取り出す。別セッションのものなら null。 */
export function loadScrollPos(sessionId: string): ScrollPos | null {
  try {
    const raw = sessionStorage.getItem(SCROLL_KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw) as ScrollPos & { sessionId: string };
    if (saved.sessionId !== sessionId) return null;
    return { scrollTop: saved.scrollTop, atBottom: saved.atBottom };
  } catch {
    return null;
  }
}

/** スクロール位置を退避する。失敗は無視する。 */
export function saveScrollPos(sessionId: string, pos: ScrollPos): void {
  try {
    sessionStorage.setItem(SCROLL_KEY, JSON.stringify({ sessionId, ...pos }));
  } catch {
    // 同上。
  }
}
