/**
 * 1on1・グループ・シナリオ共通のセッション一覧／中核 state を管理するフック。
 *
 * App コンポーネントが直接持っていた中核 state（sessions / activeSessionId / messages と
 * 競合防止用 ref）と、自己完結するセッション操作ハンドラ（新規チャット作成・新規グループ
 * チャット作成・タイトル変更）を 1 箇所へ集約する。
 *
 * 設計上の注意:
 * - 本フックは useGroupChat / useScenarioChat より「先に」呼ぶこと。返す setter
 *   （setMessages / setSessions / setActiveSessionId）と activeSessionIdRef を、それら
 *   2 フックへ依存として渡すため。
 * - 逆に他フックの出力に依存する横断的ハンドラ（handleSelectSession / handleDeleteSession）は
 *   フック同士の循環参照を避けるため本フックには含めず、App 側の配線層に残す。
 */
import {
  useCallback,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import {
  createGroupSession,
  createSession,
  updateSessionTitle,
} from "../api";
import type { ChatMessage, Session } from "../api";

/** useSessions が App から受け取る依存（本フックが所有しない共有 state の setter）。 */
interface UseSessionsDeps {
  /** エラー表示の setter。 */
  setError: (e: string | null) => void;
  /** 選択中モデルの setter（新規チャット作成時に選択キャラ/プリセットを反映する）。 */
  setSelectedModel: Dispatch<SetStateAction<string>>;
}

/** useSessions が返す中核 state・setter・ハンドラ群。 */
interface UseSessionsResult {
  /** セッション一覧（1on1・グループ）。 */
  sessions: Session[];
  /** セッション一覧の setter。 */
  setSessions: Dispatch<SetStateAction<Session[]>>;
  /** 現在アクティブなセッション ID（null = 未選択）。 */
  activeSessionId: string | null;
  /** アクティブセッション ID の setter。 */
  setActiveSessionId: Dispatch<SetStateAction<string | null>>;
  /** アクティブセッションのメッセージ一覧。 */
  messages: ChatMessage[];
  /** メッセージ一覧の setter。 */
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  /**
   * セッション切り替え競合防止用 ref。
   * 非同期ストリーミングがセッション切り替え後に state を汚染しないよう、
   * 完了時点でのアクティブセッション ID と比較するために使う。
   */
  activeSessionIdRef: MutableRefObject<string | null>;
  /**
   * 新規 1on1 チャット作成。
   * @param modelId - "{char_name}@{preset_name}" 形式のモデルID。
   */
  handleNewChat: (modelId: string) => Promise<void>;
  /**
   * 新規グループチャット作成（司会モデルはシステム設定で管理）。
   * @param participants - 参加キャラクターの model_id リスト。
   * @param maxAutoTurns - 自動進行の最大ターン数。
   */
  handleNewGroupChat: (participants: string[], maxAutoTurns: number) => Promise<void>;
  /**
   * セッションタイトル変更。
   * @param sessionId - 対象セッション ID。
   * @param newTitle - 新しいタイトル。
   */
  handleRenameSession: (sessionId: string, newTitle: string) => Promise<void>;
}

/**
 * セッション管理用フック。
 * @param deps - App が所有する共有 state の setter（error / selectedModel）。
 * @returns 中核 state・setter・自己完結セッションハンドラ群。
 */
export function useSessions(deps: UseSessionsDeps): UseSessionsResult {
  const { setError, setSelectedModel } = deps;

  /** セッション一覧（1on1・グループ）。 */
  const [sessions, setSessions] = useState<Session[]>([]);
  /** 現在アクティブなセッション ID（null = 未選択）。 */
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  /** アクティブセッションのメッセージ一覧。 */
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  /**
   * セッション切り替え競合防止用 ref。
   * 非同期処理がセッション切り替え後に state を汚染しないよう、
   * 完了時点でのアクティブセッション ID と比較するために使用する。
   * 毎レンダーで同期反映し、setState の非同期更新を待たずに最新値を読めるようにする。
   */
  const activeSessionIdRef = useRef<string | null>(activeSessionId);
  activeSessionIdRef.current = activeSessionId;

  /**
   * 新規 1on1 チャット作成。
   *
   * @param modelId - "{char_name}@{preset_name}" 形式のモデルID。
   */
  const handleNewChat = useCallback(async (modelId: string) => {
    setError(null);
    try {
      const session = await createSession(modelId);
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setMessages([]);
      // 新セッションで選んだキャラ/プリセットを selectedModel に反映する。
      // これをやらないと直前セッションの selectedModel が残り続け、
      // 最初の送信が body.model_id 経由で別キャラへ届いてしまう。
      setSelectedModel(modelId);
      // リロード後に同じセッションを復元できるようにハッシュを更新する
      window.location.hash = session.id;
    } catch (e) {
      setError(String(e));
    }
  }, [setError, setSelectedModel]);

  /** 新規グループチャット作成（司会モデルはシステム設定で管理）。 */
  const handleNewGroupChat = useCallback(async (
    participants: string[],
    maxAutoTurns: number,
  ) => {
    setError(null);
    try {
      const session = await createGroupSession(participants, maxAutoTurns, 30);
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setMessages([]);
      // リロード後に同じセッションを復元できるようにハッシュを更新する
      window.location.hash = session.id;
    } catch (e) {
      setError(String(e));
    }
  }, [setError]);

  /** セッションタイトル変更。 */
  const handleRenameSession = useCallback(async (sessionId: string, newTitle: string) => {
    try {
      const updated = await updateSessionTitle(sessionId, newTitle);
      setSessions((prev) => prev.map((s) => s.id === sessionId ? { ...s, title: updated.title } : s));
    } catch (e) {
      setError(String(e));
    }
  }, [setError]);

  return {
    sessions,
    setSessions,
    activeSessionId,
    setActiveSessionId,
    messages,
    setMessages,
    activeSessionIdRef,
    handleNewChat,
    handleNewGroupChat,
    handleRenameSession,
  };
}
