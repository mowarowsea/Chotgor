/**
 * 1on1 履歴の表示ウィンドウ（直近 N 件＋遡り読み込み）を管理するフック。
 *
 * シナリオ（useScenarioChat の applyTurnWindow / mergeTurnWindow / loadOlderScenarioTurns）
 * と同じ考え方を 1on1 へ横展開したもの。スクロール機構（センチネル・位置補正）は
 * `useInfiniteScrollTop` を両者で共有し、取得層だけがここにある
 * （カーソルが `turn_index` ではなくメッセージ ID、API も別のため）。
 *
 * 1on1 に枝（variant）による前方の巻き戻しは無いので、ウィンドウの反映は
 * 「マージ」1 本で足りる: 取得した直近ウィンドウの先頭 ID が表示中にあれば、それより
 * 手前の表示中メッセージ（遡って読み込んだぶん・スナップショットから描いたぶん）を残す。
 */
import {
  useCallback,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import { fetchOlderMessages, MESSAGE_PAGE_SIZE } from "../api";
import type { ChatMessage, SessionDetail } from "../api";

/** useMessageWindow が App から受け取る依存。 */
interface UseMessageWindowDeps {
  /** 現在アクティブなセッション ID。 */
  activeSessionId: string | null;
  /** セッション切り替え競合防止用 ref（着弾時点での active ID と比較する）。 */
  activeSessionIdRef: MutableRefObject<string | null>;
  /** メッセージ一覧とその setter（所有は useSessions）。 */
  messages: ChatMessage[];
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  /** reasoning / ログ ID のマップ setter。遡って読んだぶんも表示用マップへ足す。 */
  setReasoningMap: Dispatch<SetStateAction<Record<string, string>>>;
  setMsgLogIds: Dispatch<SetStateAction<Record<string, string>>>;
  /** エラー表示の setter。 */
  setError: (e: string | null) => void;
}

/** useMessageWindow が返す state・ハンドラ群。 */
interface UseMessageWindowResult {
  /** 表示中ウィンドウより過去のメッセージがまだ残っているか。 */
  hasOlderMessages: boolean;
  /** 遡り読み込みの実行中フラグ。 */
  loadingOlderMessages: boolean;
  /**
   * 取得したセッション詳細（直近ウィンドウ）を表示へ反映する。
   * 表示中の先頭側を残すマージを行い、reasoning / ログ ID もウィンドウぶんを足す。
   */
  applyDetail: (detail: SessionDetail) => void;
  /** 表示ウィンドウを 1 ページぶん過去へ伸ばす。 */
  loadOlderMessages: () => Promise<void>;
  /** スナップショットから描いた直後など、ウィンドウの上端判定だけを立てる。 */
  setHasOlderMessages: Dispatch<SetStateAction<boolean>>;
  /** セッション切替・削除時にウィンドウ状態を初期化する。 */
  resetWindow: () => void;
}

/**
 * メッセージ配列から reasoning / log_message_id を表示用マップへ取り出す。
 * セッション読込時にしか復元していないマップへ、遡って読んだぶんを足すために使う。
 */
function extractMaps(messages: ChatMessage[]): {
  reasonings: Record<string, string>;
  logIds: Record<string, string>;
} {
  const reasonings: Record<string, string> = {};
  const logIds: Record<string, string> = {};
  for (const msg of messages) {
    if (msg.reasoning) reasonings[msg.id] = msg.reasoning;
    if (msg.log_message_id) logIds[msg.id] = msg.log_message_id;
  }
  return { reasonings, logIds };
}

/**
 * 直近ウィンドウ `fresh` を表示中 `prev` へマージする。
 *
 * `fresh` の先頭 ID が `prev` にあれば、それより手前の `prev` を残して後ろを `fresh` で
 * 置き換える（遡って読み込んだ履歴を、再取得のたびに畳まない）。無ければ丸ごと差し替え
 * （別セッション・初回・表示中より新しいメッセージだけが返ったとき）。
 */
export function mergeMessageWindow(prev: ChatMessage[], fresh: ChatMessage[]): ChatMessage[] {
  if (fresh.length === 0) return fresh;
  const idx = prev.findIndex((m) => m.id === fresh[0].id);
  if (idx <= 0) return fresh;
  return [...prev.slice(0, idx), ...fresh];
}

/**
 * 1on1 履歴ウィンドウ用フック。
 * @param deps - App が所有する共有 state の setter とセッション情報。
 */
export function useMessageWindow(deps: UseMessageWindowDeps): UseMessageWindowResult {
  const {
    activeSessionId,
    activeSessionIdRef,
    messages,
    setMessages,
    setReasoningMap,
    setMsgLogIds,
    setError,
  } = deps;

  const [hasOlderMessages, setHasOlderMessages] = useState(false);
  const [loadingOlderMessages, setLoadingOlderMessages] = useState(false);
  /** messages の最新値を setState のクロージャ外で読むための ref。 */
  const messagesRef = useRef<ChatMessage[]>(messages);
  messagesRef.current = messages;

  const resetWindow = useCallback(() => {
    setHasOlderMessages(false);
    setLoadingOlderMessages(false);
  }, []);

  const applyDetail = useCallback((detail: SessionDetail) => {
    const prev = messagesRef.current;
    const sameSession = prev.length > 0 && prev[0].session_id === detail.id;
    const merged = sameSession ? mergeMessageWindow(prev, detail.messages) : detail.messages;
    // 手前を残した（＝取得ウィンドウより古い表示があった）なら、上端判定は表示中の
    // 先頭に対するもので、ウィンドウ先頭に対するサーバの答えとは別物。元の値を保つ。
    const keptPrefix = merged.length > detail.messages.length;
    setMessages(merged);
    if (!keptPrefix) setHasOlderMessages(detail.has_older_messages ?? false);
    const { reasonings, logIds } = extractMaps(detail.messages);
    if (sameSession) {
      setReasoningMap((m) => ({ ...m, ...reasonings }));
      setMsgLogIds((m) => ({ ...m, ...logIds }));
    } else {
      setReasoningMap(reasonings);
      setMsgLogIds(logIds);
    }
  }, [setMessages, setReasoningMap, setMsgLogIds]);

  const loadOlderMessages = useCallback(async () => {
    const sessionId = activeSessionId;
    if (!sessionId || loadingOlderMessages || !hasOlderMessages) return;
    const oldest = messagesRef.current[0];
    // 楽観バブル（optimistic-…）は DB に無いのでカーソルにできない。先頭に来ることはないが保険。
    if (!oldest || oldest.id.startsWith("optimistic-")) return;
    setLoadingOlderMessages(true);
    try {
      const { messages: older, has_older } = await fetchOlderMessages(
        sessionId, oldest.id, MESSAGE_PAGE_SIZE,
      );
      // 待っている間にセッションが切り替わっていたら、別セッションの先頭へ積まない。
      if (activeSessionIdRef.current !== sessionId) return;
      setMessages((prev) => [...older, ...prev]);
      setHasOlderMessages(has_older);
      const { reasonings, logIds } = extractMaps(older);
      setReasoningMap((m) => ({ ...m, ...reasonings }));
      setMsgLogIds((m) => ({ ...m, ...logIds }));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingOlderMessages(false);
    }
  }, [
    activeSessionId,
    activeSessionIdRef,
    loadingOlderMessages,
    hasOlderMessages,
    setMessages,
    setReasoningMap,
    setMsgLogIds,
    setError,
  ]);

  return {
    hasOlderMessages,
    loadingOlderMessages,
    applyDetail,
    loadOlderMessages,
    setHasOlderMessages,
    resetWindow,
  };
}
