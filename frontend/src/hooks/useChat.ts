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
  fetchCharacters,
  fetchSessions,
  streamMessage,
} from "../api";
import type {
  Attachment,
  Character,
  ChatMessage,
  Session,
  StreamEvent,
} from "../api";
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
  /** メッセージ一覧の setter。 */
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  /** セッション一覧の setter。 */
  setSessions: Dispatch<SetStateAction<Session[]>>;
  /**
   * キャラクター一覧の setter。
   * ターン中にキャラクター本人がツールで自分の状態を変えることがある
   * （visit_user の対面モード ON など）。characters は初期マウント時にしか
   * 取得していないため、done のたびに取り直さないと画面が古い値を持ち続ける。
   */
  setCharacters: Dispatch<SetStateAction<Character[]>>;
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
    attachments?: Attachment[],
    modelId?: string,
  ) => Promise<void>;
  /** 編集・再生成: fromMessageId 以降を削除して再送する。 */
  handleRetry: (
    fromMessageId: string,
    content: string,
    attachments?: Attachment[],
  ) => Promise<void>;
  /** 末尾ユーザメッセージの削除（再送はしない）。 */
  handleDeleteMessage: (messageId: string) => Promise<void>;
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
    setMessages,
    setSessions,
    setCharacters,
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
    attachments: Attachment[] = [],
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
      attachments: attachments.length > 0 ? attachments : undefined,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUserMsg]);

    let accumulatedReasoning = "";
    // done イベントを onEvent 内で同期的に処理しきれない（fetchSessions が await を要する）
    // ため、最終 done event を変数に退避してループ後に await する。
    type DoneEvent = Extract<StreamEvent, { type: "done" }>;
    let pendingDone: DoneEvent | null = null;
    await consumeStream<StreamEvent>({
      stream: streamMessage(sessionId, content, attachments.map((a) => a.id), modelId),
      onEvent: (event) => {
        if (event.type === "chunk") {
          setStreamingContent((prev) => (prev ?? "") + event.content);
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
      // セッション（タイトル・current_bg_label 等）とキャラクター（対面モード等）を
      // まとめて取り直す。キャラ側は本人のツール実行で変わりうるが SSE では流れてこない。
      // 失敗しても会話表示は成立するため、キャラ側は best-effort（握って続行）。
      const [updated] = await Promise.all([
        fetchSessions(),
        fetchCharacters()
          .then(setCharacters)
          .catch(() => undefined),
      ]);
      setSessions(updated);
    }
  }, [
    activeSessionIdRef,
    setMessages,
    setSessions,
    setCharacters,
    setError,
    setElapsedMap,
    setMsgLogIds,
  ]);

  /**
   * ユーザメッセージ編集 / キャラクター応答再生成の共通ハンドラ。
   * fromMessageId 以降をDBから削除し、content でストリームを再送する。
   * 再生成の場合は fromMessageId = 直前ユーザメッセージのID、content = そのメッセージ本文。
   * attachments = 再送する添付リスト（再生成時は元メッセージの添付を引き継ぐ）。
   */
  const handleRetry = useCallback(async (
    fromMessageId: string,
    content: string,
    attachments: Attachment[] = [],
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
      await doStream(activeSessionId, content, attachments, selectedModel || undefined);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [activeSessionId, sending, selectedModel, doStream, setError, setSending, setMessages]);

  /**
   * ユーザメッセージの削除。
   *
   * 削除 API は「指定メッセージ以降」しか持たないため、呼び出し側は末尾の
   * ユーザメッセージにだけこのハンドラを渡すこと（それ以外へ渡すと後続の
   * やり取りごと消える）。UI 上はセッション末尾のバブルにしかゴミ箱を出さない。
   * 再ストリームは行わない — 送ってしまった発話をなかったことにするための操作。
   */
  const handleDeleteMessage = useCallback(async (messageId: string) => {
    if (!activeSessionId || sending) return;
    setError(null);
    try {
      await deleteMessagesFrom(activeSessionId, messageId);
      setMessages((prev) => {
        const idx = prev.findIndex((m) => m.id === messageId);
        return idx >= 0 ? prev.slice(0, idx) : prev;
      });
    } catch (e) {
      setError(String(e));
    }
  }, [activeSessionId, sending, setError, setMessages]);

  return {
    streamingContent,
    streamingReasoning,
    reasoningMap,
    setReasoningMap,
    resetStreamingState,
    doStream,
    handleRetry,
    handleDeleteMessage,
  };
}
