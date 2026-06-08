/**
 * グループチャットの状態管理・ストリーミング送受信を担うフック。
 *
 * App コンポーネントに散在していた group 系 state（待機キャラ・ストリーミング内容・
 * 思考ブロック・reasoningMap・ユーザターン判定・司会エラー）と、その操作ハンドラ
 * （_doGroupStream / retry / skip / retryDirector / requestCharacter）を 1 箇所へ集約する。
 *
 * 1on1・シナリオと共有する state（messages / sessions / sending / activeSessionId 等）は
 * フックに閉じ込めず、引数で setter を受け取る形にして所有権を App 側へ残す。
 */
import {
  useCallback,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import {
  deleteMessagesFrom,
  fetchSession,
  fetchSessions,
  streamGroupMessage,
} from "../api";
import type { ChatMessage, GroupStreamEvent, Session } from "../api";

/** useGroupChat が App から受け取る依存（共有 state の setter 群とセッション情報）。 */
interface UseGroupChatDeps {
  /** 現在アクティブなセッション ID。 */
  activeSessionId: string | null;
  /** 送信中フラグ（多重送信防止に使う）。 */
  sending: boolean;
  /** セッション切り替え競合防止用 ref（完了時点での active ID と比較する）。 */
  activeSessionIdRef: MutableRefObject<string | null>;
  /** メッセージ一覧の setter。 */
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  /** セッション一覧の setter。 */
  setSessions: Dispatch<SetStateAction<Session[]>>;
  /** エラー表示の setter。 */
  setError: (e: string | null) => void;
  /** 送信中フラグの setter。 */
  setSending: Dispatch<SetStateAction<boolean>>;
  /** 経過時間マップの setter。 */
  setElapsedMap: Dispatch<SetStateAction<Record<string, number>>>;
}

/** useGroupChat が返す state・setter・ハンドラ群。 */
interface UseGroupChatResult {
  /** 応答待機中・ストリーミング中のキャラクター名（null = なし）。 */
  groupWaitingCharacter: string | null;
  /** ストリーミング中の応答テキスト。 */
  groupStreamingContent: string | null;
  /** ストリーミング中の思考ブロック・想起記憶テキスト。 */
  groupStreamingReasoning: string | null;
  /** メッセージID → reasoning テキストのマップ。 */
  groupReasoningMap: Record<string, string>;
  /** ユーザターン待ち状態か（スキップボタン表示制御に使う）。 */
  isGroupUserTurn: boolean;
  /** 直近ターンで司会がエラーになったか（司会再試行ボタン表示に使う）。 */
  groupDirectorErrored: boolean;
  /** reasoningMap の setter（セッション選択時の復元に使う）。 */
  setGroupReasoningMap: Dispatch<SetStateAction<Record<string, string>>>;
  /** ユーザターン判定の setter（セッション選択時の復元に使う）。 */
  setIsGroupUserTurn: Dispatch<SetStateAction<boolean>>;
  /** ストリーミング系 state を初期化する（セッション切り替え時に呼ぶ）。 */
  resetStreamingState: () => void;
  /** グループストリーミング送信の実体（handleSend からも呼ばれる）。 */
  doGroupStream: (
    sessionId: string,
    content: string,
    imageIds?: string[],
    skip?: boolean,
    targetCharacter?: string | null,
  ) => Promise<void>;
  /** 編集・再生成: fromMessageId 以降を削除して再送する。 */
  handleGroupRetry: (
    fromMessageId: string,
    content: string,
    imageIds?: string[],
  ) => Promise<void>;
  /** ユーザターンスキップ: ユーザ発話を保存せず司会へ委譲する。 */
  handleGroupSkip: () => Promise<void>;
  /** 司会の手動再試行。 */
  handleGroupRetryDirector: () => Promise<void>;
  /** 参加者を手動指名して発言させる。 */
  handleGroupRequestCharacter: (charName: string) => Promise<void>;
}

/**
 * グループチャット用フック。
 * @param deps - App が所有する共有 state の setter とセッション情報。
 * @returns グループ専用 state・setter・ハンドラ群。
 */
export function useGroupChat(deps: UseGroupChatDeps): UseGroupChatResult {
  const {
    activeSessionId,
    sending,
    activeSessionIdRef,
    setMessages,
    setSessions,
    setError,
    setSending,
    setElapsedMap,
  } = deps;

  /** 応答待機中・ストリーミング中のキャラクター名（null = なし）。 */
  const [groupWaitingCharacter, setGroupWaitingCharacter] = useState<string | null>(null);
  /** ストリーミング中の応答テキスト。 */
  const [groupStreamingContent, setGroupStreamingContent] = useState<string | null>(null);
  /** ストリーミング中の思考ブロック・想起記憶テキスト。 */
  const [groupStreamingReasoning, setGroupStreamingReasoning] = useState<string | null>(null);
  /** メッセージID → reasoning テキストのマップ。 */
  const [groupReasoningMap, setGroupReasoningMap] = useState<Record<string, string>>({});
  /** ユーザターン待ち状態か。 */
  const [isGroupUserTurn, setIsGroupUserTurn] = useState(false);
  /** 直近ターンで司会がエラーになったか。 */
  const [groupDirectorErrored, setGroupDirectorErrored] = useState(false);

  /** ストリーミング系 state を初期化する（セッション切り替え時に呼ぶ）。 */
  const resetStreamingState = useCallback(() => {
    setGroupWaitingCharacter(null);
    setGroupStreamingContent(null);
    setGroupStreamingReasoning(null);
    setIsGroupUserTurn(false);
    setGroupDirectorErrored(false);
  }, []);

  /**
   * グループチャットのメッセージ送信実装。
   * SSE受信中はキャラクター名を waitingCharacter で表示し、
   * 受信完了後に全メッセージをサーバーから再取得して確定する。
   *
   * @param skip - true の場合、ユーザメッセージを保存せず司会へ直接ターンを委譲する（ユーザターンスキップ）。
   * @param targetCharacter - 指定した場合、司会を介さずそのキャラクターを手動指名して発言させる。
   */
  const doGroupStream = useCallback(async (
    sessionId: string,
    content: string,
    imageIds: string[] = [],
    skip = false,
    targetCharacter: string | null = null,
  ) => {
    setError(null);
    setSending(true);
    setIsGroupUserTurn(false);
    setGroupDirectorErrored(false);
    setGroupWaitingCharacter(null);

    // キャラクターごとのリクエスト〜応答完了の経過時間を計測する開始時刻。
    // character_start で更新し character_done で差分を取る（1ターン中に複数キャラが続けて発話するケースに対応）。
    let charStartedAt: number | null = null;

    const optimisticId = `optimistic-${Date.now()}`;
    // スキップ時・手動指名時はユーザメッセージを保存しないため楽観的表示も行わない
    if (!skip && !targetCharacter) {
      const optimisticUserMsg: ChatMessage = {
        id: optimisticId,
        session_id: sessionId,
        role: "user",
        content,
        images: imageIds.length > 0 ? imageIds : undefined,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, optimisticUserMsg]);
    }

    try {
      for await (const event of streamGroupMessage(sessionId, content, imageIds, skip, targetCharacter)) {
        const ev = event as GroupStreamEvent;
        if (ev.type === "user_saved") {
          // optimisticメッセージを確定済みユーザーメッセージで差し替える
          setMessages((prev) => prev.map((m) => m.id === optimisticId ? ev.message : m));
        } else if (ev.type === "character_start") {
          // キャラクター応答開始：スピナーを表示し、前のストリーミング内容をクリアする
          setGroupWaitingCharacter(ev.character);
          setGroupStreamingContent(null);
          setGroupStreamingReasoning(null);
          // 経過時間計測の開始時刻を更新する
          charStartedAt = performance.now();
        } else if (ev.type === "character_reasoning") {
          // 思考ブロック・想起記憶をリアルタイム表示する
          setGroupStreamingReasoning((prev) => (prev ?? "") + ev.content);
        } else if (ev.type === "character_chunk") {
          // 応答テキストを表示する
          setGroupStreamingContent(ev.content);
        } else if (ev.type === "character_done") {
          // キャラクター応答確定：メッセージリストに追加してストリーミング状態をクリアする
          if (ev.message.reasoning) {
            setGroupReasoningMap((prev) => ({ ...prev, [ev.message.id]: ev.message.reasoning! }));
          }
          // 経過時間を記録する（character_start からの差分）
          if (charStartedAt !== null) {
            const elapsed = performance.now() - charStartedAt;
            setElapsedMap((prev) => ({ ...prev, [ev.message.id]: elapsed }));
            charStartedAt = null;
          }
          setMessages((prev) => [...prev, ev.message]);
          setGroupStreamingContent(null);
          setGroupStreamingReasoning(null);
          // waitingCharacter は次の character_start か user_turn まで維持する
        } else if (ev.type === "character_angle_switched") {
          // キャラクターがプリセットを切り替えた。
          // バックエンドが group_config を永続化済みのため、ストリーム終了後の finally で
          // セッション再取得すれば参加者のプリセット情報が更新される。
        } else if (ev.type === "director_error") {
          // 司会エラー：エラー表示しユーザターンへ戻す。司会再試行・手動指名で復帰可能。
          setGroupWaitingCharacter(null);
          setGroupStreamingContent(null);
          setGroupStreamingReasoning(null);
          setError(ev.message);
          setGroupDirectorErrored(true);
          setIsGroupUserTurn(true);
          break;
        } else if (ev.type === "user_turn" || ev.type === "done") {
          setGroupWaitingCharacter(null);
          setGroupStreamingContent(null);
          setGroupStreamingReasoning(null);
          if (ev.type === "user_turn") setIsGroupUserTurn(true);
          break;
        } else if (ev.type === "error") {
          setGroupWaitingCharacter(null);
          setGroupStreamingContent(null);
          setGroupStreamingReasoning(null);
          setError(ev.message);
          break;
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      // サーバーから最新状態を取得して確定する（セッションが切り替わっていなければ）
      try {
        if (sessionId === activeSessionIdRef.current) {
          const detail = await fetchSession(sessionId);
          setMessages(detail.messages);
          const updated = await fetchSessions();
          setSessions(updated);
        }
      } catch {
        // 取得失敗は無視する
      }
      setGroupWaitingCharacter(null);
      setGroupStreamingContent(null);
      setGroupStreamingReasoning(null);
      setSending(false);
    }
  }, [activeSessionIdRef, setMessages, setSessions, setError, setSending, setElapsedMap]);

  /**
   * グループチャットの編集・再生成共通ハンドラ。
   * fromMessageId 以降をDBから削除し、content でグループストリームを再送する。
   * imageIds には再送する画像IDリストを渡す（再生成時は元メッセージの画像を引き継ぐ）。
   */
  const handleGroupRetry = useCallback(async (fromMessageId: string, content: string, imageIds: string[] = []) => {
    if (!activeSessionId || sending) return;
    setError(null);
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === fromMessageId);
      return idx >= 0 ? prev.slice(0, idx) : prev;
    });
    try {
      await deleteMessagesFrom(activeSessionId, fromMessageId);
    } catch (e) {
      setError(String(e));
      return;
    }
    doGroupStream(activeSessionId, content, imageIds);
  }, [activeSessionId, sending, doGroupStream, setError, setMessages]);

  /**
   * グループチャットのユーザターンスキップ。
   * ユーザメッセージを保存せず、司会へ直接ターンを委譲する。
   */
  const handleGroupSkip = useCallback(async () => {
    if (!activeSessionId || sending) return;
    doGroupStream(activeSessionId, "", [], true);
  }, [activeSessionId, sending, doGroupStream]);

  /**
   * グループチャットの司会を手動で再試行する。
   * 司会エラー後にユーザがもう一度司会へ次発言者の判断を依頼する。
   */
  const handleGroupRetryDirector = useCallback(async () => {
    if (!activeSessionId || sending) return;
    doGroupStream(activeSessionId, "", [], true);
  }, [activeSessionId, sending, doGroupStream]);

  /**
   * グループチャットで任意の参加者を手動指名して発言させる。
   * 司会を介さず指定キャラクターに直接リクエストする（司会エラー時の代替手段）。
   */
  const handleGroupRequestCharacter = useCallback(async (charName: string) => {
    if (!activeSessionId || sending) return;
    doGroupStream(activeSessionId, "", [], false, charName);
  }, [activeSessionId, sending, doGroupStream]);

  return {
    groupWaitingCharacter,
    groupStreamingContent,
    groupStreamingReasoning,
    groupReasoningMap,
    isGroupUserTurn,
    groupDirectorErrored,
    setGroupReasoningMap,
    setIsGroupUserTurn,
    resetStreamingState,
    doGroupStream,
    handleGroupRetry,
    handleGroupSkip,
    handleGroupRetryDirector,
    handleGroupRequestCharacter,
  };
}
