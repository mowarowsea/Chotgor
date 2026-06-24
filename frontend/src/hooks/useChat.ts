/**
 * 1on1チャットの状態管理・ストリーミング送受信を担うフック。
 *
 * App コンポーネントに残っていた 1on1 系 state（ストリーミング内容・思考ブロック・
 * reasoningMap）と、その操作ハンドラ（doStream / handleRetry）を 1 箇所へ集約する。
 *
 * 1on1・シナリオと共有する state（messages / sessions / sending / selectedModel 等）は
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
  fetchSessions,
  streamMessage,
} from "../api";
import type { ChatMessage, Session, StreamEvent } from "../api";
import { consumeStream } from "./streamingUtils";

/** useChat が App から受け取る依存（共有 state の setter 群とセッション情報）。 */
interface UseChatDeps {
  /** 現在アクティブなセッション ID。 */
  activeSessionId: string | null;
  /** 送信中フラグ（多重送信防止に使う）。 */
  sending: boolean;
  /** セッション切り替え競合防止用 ref（完了時点での active ID と比較する）。 */
  activeSessionIdRef: MutableRefObject<string | null>;
  /** 選択中モデル ID（再生成時のリクエストモデルに使う）。 */
  selectedModel: string;
  /** 選択中モデルの setter（switch_angle 完了時に切り替え先へ更新する）。 */
  setSelectedModel: Dispatch<SetStateAction<string>>;
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
  /** char_msg_id → log_message_id マップの setter（デバッグログ紐付け用）。 */
  setMsgLogIds: Dispatch<SetStateAction<Record<string, string>>>;
}

/** useChat が返す state・setter・ハンドラ群。 */
interface UseChatResult {
  /** ストリーミング中の応答テキスト。 */
  streamingContent: string | null;
  /** ストリーミング中の思考ブロック・想起記憶テキスト。 */
  streamingReasoning: string | null;
  /** メッセージID → reasoning テキストのマップ。 */
  reasoningMap: Record<string, string>;
  /** reasoningMap の setter（セッション選択時の復元に使う）。 */
  setReasoningMap: Dispatch<SetStateAction<Record<string, string>>>;
  /** ストリーミング系 state を初期化する（セッション切り替え時に呼ぶ）。 */
  resetStreamingState: () => void;
  /**
   * 1on1 ストリーミング送信の実体（handleSend / handleRetry から呼ばれる）。
   * 楽観的ユーザメッセージ表示 + SSE受信を行う。
   */
  doStream: (
    sessionId: string,
    content: string,
    imageIds?: string[],
    modelId?: string,
  ) => Promise<void>;
  /** 編集・再生成: fromMessageId 以降を削除して再送する。 */
  handleRetry: (
    fromMessageId: string,
    content: string,
    imageIds?: string[],
  ) => Promise<void>;
}

/**
 * 1on1チャット用フック。
 * @param deps - App が所有する共有 state の setter とセッション情報。
 * @returns 1on1 専用 state・setter・ハンドラ群。
 */
export function useChat(deps: UseChatDeps): UseChatResult {
  const {
    activeSessionId,
    sending,
    activeSessionIdRef,
    selectedModel,
    setSelectedModel,
    setMessages,
    setSessions,
    setError,
    setSending,
    setElapsedMap,
    setMsgLogIds,
  } = deps;

  /** ストリーミング中の応答テキスト。 */
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  /** ストリーミング中の思考ブロック・想起記憶テキスト。 */
  const [streamingReasoning, setStreamingReasoning] = useState<string | null>(null);
  /** メッセージID → reasoning テキストのマップ。 */
  const [reasoningMap, setReasoningMap] = useState<Record<string, string>>({});

  /** ストリーミング系 state を初期化する（セッション切り替え時に呼ぶ）。 */
  const resetStreamingState = useCallback(() => {
    setStreamingContent(null);
    setStreamingReasoning(null);
  }, []);

  /**
   * ストリーミング送信の共通実装。楽観的ユーザメッセージ表示 + SSE受信を行う。
   * handleSend / handleRetry の両方から呼ばれる。
   */
  const doStream = useCallback(async (
    sessionId: string,
    content: string,
    imageIds: string[] = [],
    modelId?: string,
  ) => {
    setError(null);
    setStreamingContent("");
    setStreamingReasoning(null);
    // モデルリクエスト〜応答完了までの経過時間を計測する開始時刻。
    const streamStartedAt = performance.now();

    const optimisticUserMsg: ChatMessage = {
      id: `optimistic-${Date.now()}`,
      session_id: sessionId,
      role: "user",
      content,
      images: imageIds.length > 0 ? imageIds : undefined,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUserMsg]);

    let accumulatedReasoning = "";
    // done イベントを onEvent 内で同期的に処理しきれない（fetchSessions が await を要する）
    // ため、最終 done event を変数に退避してループ後に await する。
    type DoneEvent = Extract<StreamEvent, { type: "done" }>;
    let pendingDone: DoneEvent | null = null;
    await consumeStream<StreamEvent>({
      stream: streamMessage(sessionId, content, imageIds, modelId),
      onEvent: (event) => {
        if (event.type === "chunk") {
          setStreamingContent((prev) => (prev ?? "") + event.content);
        } else if (event.type === "clear") {
          // switch_angle 発動: 第1プロバイダーの表示をクリアして第2プロバイダーを待つ
          setStreamingContent("");
          accumulatedReasoning = "";
          setStreamingReasoning(null);
        } else if (event.type === "angle_switched") {
          // switch_angle 完了: selectedModel を切り替え先に更新する。
          // これを行わないと次ターン以降も元のプリセットでリクエストされ続け、
          // 切り替えが1ターン限りで消えてしまう。
          setSelectedModel(event.model_id);
        } else if (event.type === "reasoning") {
          accumulatedReasoning += event.content;
          setStreamingReasoning(accumulatedReasoning);
        } else if (event.type === "done") {
          pendingDone = event;
          return false; // ループ終了
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      },
      onError: (e) => {
        setStreamingContent(null);
        setStreamingReasoning(null);
        setMessages((prev) => prev.filter((m) => m.id !== optimisticUserMsg.id));
        setError(String(e));
      },
    });

    // done イベント後処理: セッションが切り替わっていたら state を汚染しない。
    // setReasoningMap 等の setter コールバック内で参照する値は、まずローカル変数に
    // 取り出してから渡す（closure 越しに非 null narrowing が失われる回避）。
    if (pendingDone !== null) {
      const doneEvent = pendingDone as DoneEvent;
      const charMsgId = doneEvent.character_message.id;
      const doneLogId = doneEvent.log_message_id ?? null;
      const userMessage = doneEvent.user_message;
      const characterMessage = doneEvent.character_message;

      if (sessionId !== activeSessionIdRef.current) return;
      if (accumulatedReasoning) {
        const reasoning = accumulatedReasoning;
        setReasoningMap((prev) => ({ ...prev, [charMsgId]: reasoning }));
      }
      if (doneLogId) {
        setMsgLogIds((prev) => ({ ...prev, [charMsgId]: doneLogId }));
      }
      setElapsedMap((prev) => ({
        ...prev,
        [charMsgId]: performance.now() - streamStartedAt,
      }));
      setStreamingContent(null);
      setStreamingReasoning(null);
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== optimisticUserMsg.id),
        userMessage,
        characterMessage,
      ]);
      const updated = await fetchSessions();
      setSessions(updated);
    }
  }, [
    activeSessionIdRef,
    setSelectedModel,
    setMessages,
    setSessions,
    setError,
    setElapsedMap,
    setMsgLogIds,
  ]);

  /**
   * ユーザメッセージ編集 / キャラクター応答再生成の共通ハンドラ。
   * fromMessageId 以降をDBから削除し、content でストリームを再送する。
   * 再生成の場合は fromMessageId = 直前ユーザメッセージのID、content = そのメッセージ本文。
   * imageIds = 再送する画像IDリスト（再生成時は元メッセージの画像を引き継ぐ）。
   */
  const handleRetry = useCallback(async (
    fromMessageId: string,
    content: string,
    imageIds: string[] = [],
  ) => {
    if (!activeSessionId || sending) return;
    setSending(true);
    setError(null);
    // ローカル状態を即時切り詰めてUIを反映する
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === fromMessageId);
      return idx >= 0 ? prev.slice(0, idx) : prev;
    });
    try {
      await deleteMessagesFrom(activeSessionId, fromMessageId);
      await doStream(activeSessionId, content, imageIds, selectedModel || undefined);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [activeSessionId, sending, selectedModel, doStream, setError, setSending, setMessages]);

  return {
    streamingContent,
    streamingReasoning,
    reasoningMap,
    setReasoningMap,
    resetStreamingState,
    doStream,
    handleRetry,
  };
}
