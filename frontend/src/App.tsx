/**
 * アプリルートコンポーネント。
 * セッション管理・モデル取得・メッセージ送受信のロジックを担当する。
 */
import { useEffect, useState, useCallback } from "react";
import {
  fetchModels,
  fetchSessions,
  fetchSession,
  createSession,
  deleteSession,
  deleteMessagesFrom,
  streamMessage,
  fetchUserName,
} from "./api";
import type { Model, Session, ChatMessage, StreamEvent } from "./api";
import Sidebar from "./components/Sidebar";
import ChatView from "./components/ChatView";

/** アプリ全体のルートコンポーネント。 */
export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  /** ストリーミング中の思考ブロック・想起記憶テキスト（null = なし） */
  const [streamingReasoning, setStreamingReasoning] = useState<string | null>(null);
  /** 完了済みメッセージIDに紐付いた reasoning テキスト。ページリロードまで保持する。 */
  const [reasoningMap, setReasoningMap] = useState<Record<string, string>>({});
  const [userName, setUserName] = useState("ユーザ");
  const [error, setError] = useState<string | null>(null);

  /** アクティブセッションのキャラクター名を model_id から抽出する。 */
  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const characterName = activeSession?.model_id.split("@")[0] ?? "キャラクター";

  /** 初期データ取得。 */
  useEffect(() => {
    Promise.all([fetchModels(), fetchSessions(), fetchUserName()])
      .then(([m, s, u]) => {
        setModels(m);
        setSessions(s);
        setUserName(u);
      })
      .catch((e) => setError(String(e)));
  }, []);

  /** セッション選択時にメッセージ一覧を取得し、reasoningMap を復元する。 */
  const handleSelectSession = useCallback(async (sessionId: string) => {
    setActiveSessionId(sessionId);
    setError(null);
    try {
      const detail = await fetchSession(sessionId);
      setMessages(detail.messages);
      // DBに保存された reasoning をメッセージIDに紐付けて復元する
      const restored: Record<string, string> = {};
      for (const msg of detail.messages) {
        if (msg.reasoning) {
          restored[msg.id] = msg.reasoning;
        }
      }
      setReasoningMap(restored);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  /** 新規チャット作成。 */
  const handleNewChat = useCallback(async (modelId: string) => {
    setError(null);
    try {
      const session = await createSession(modelId);
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setMessages([]);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  /** セッション削除。 */
  const handleDeleteSession = useCallback(async (sessionId: string) => {
    setError(null);
    try {
      await deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
    } catch (e) {
      setError(String(e));
    }
  }, [activeSessionId]);

  /**
   * ストリーミング送信の共通実装。楽観的ユーザメッセージ表示 + SSE受信を行う。
   * handleSend / handleRetry の両方から呼ばれる。
   */
  const _doStream = useCallback(async (sessionId: string, content: string) => {
    setError(null);
    setStreamingContent("");
    setStreamingReasoning(null);

    const optimisticUserMsg: ChatMessage = {
      id: `optimistic-${Date.now()}`,
      session_id: sessionId,
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUserMsg]);

    let accumulatedReasoning = "";
    try {
      for await (const event of streamMessage(sessionId, content)) {
        if (event.type === "chunk") {
          setStreamingContent((prev) => (prev ?? "") + event.content);
        } else if (event.type === "reasoning") {
          accumulatedReasoning += event.content;
          setStreamingReasoning(accumulatedReasoning);
        } else if (event.type === "done") {
          if (accumulatedReasoning) {
            setReasoningMap((prev) => ({
              ...prev,
              [event.character_message.id]: accumulatedReasoning,
            }));
          }
          setStreamingContent(null);
          setStreamingReasoning(null);
          setMessages((prev) => [
            ...prev.filter((m) => m.id !== optimisticUserMsg.id),
            event.user_message,
            event.character_message,
          ]);
          const updated = await fetchSessions();
          setSessions(updated);
        } else if (event.type === "error") {
          throw new Error((event as StreamEvent & { type: "error" }).message);
        }
      }
    } catch (e) {
      setStreamingContent(null);
      setStreamingReasoning(null);
      setMessages((prev) => prev.filter((m) => m.id !== optimisticUserMsg.id));
      setError(String(e));
    }
  }, []);

  /** メッセージ送信。ユーザーバルーン即時表示 + SSEストリーミングでキャラクター応答を表示する。 */
  const handleSend = useCallback(async (content: string) => {
    if (!activeSessionId) return;
    setSending(true);
    try {
      await _doStream(activeSessionId, content);
    } finally {
      setSending(false);
    }
  }, [activeSessionId, _doStream]);

  /**
   * ユーザメッセージ編集 / キャラクター応答再生成の共通ハンドラ。
   * fromMessageId 以降をDBから削除し、content でストリームを再送する。
   * 再生成の場合は fromMessageId = 直前ユーザメッセージのID、content = そのメッセージ本文。
   */
  const handleRetry = useCallback(async (fromMessageId: string, content: string) => {
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
      await _doStream(activeSessionId, content);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [activeSessionId, sending, _doStream]);

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950 text-zinc-100">
      <Sidebar
        models={models}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
      />

      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {/* エラーバナー */}
        {error && (
          <div className="bg-red-900/60 text-red-200 text-sm px-4 py-2 flex justify-between items-center">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-red-300 hover:text-red-100">✕</button>
          </div>
        )}

        {activeSessionId ? (
          <ChatView
            messages={messages}
            characterName={characterName}
            userName={userName}
            sending={sending}
            streamingContent={streamingContent}
            streamingReasoning={streamingReasoning}
            reasoningMap={reasoningMap}
            onSend={handleSend}
            onRetry={handleRetry}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center text-zinc-500 text-sm">
            <p>左のサイドバーからチャットを選択するか、新規チャットを作成してください</p>
          </div>
        )}
      </main>
    </div>
  );
}
