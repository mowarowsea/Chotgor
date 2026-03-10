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
  sendMessage,
  fetchUserName,
} from "./api";
import type { Model, Session, ChatMessage } from "./api";
import Sidebar from "./components/Sidebar";
import ChatView from "./components/ChatView";

/** アプリ全体のルートコンポーネント。 */
export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);
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

  /** セッション選択時にメッセージ一覧を取得する。 */
  const handleSelectSession = useCallback(async (sessionId: string) => {
    setActiveSessionId(sessionId);
    setError(null);
    try {
      const detail = await fetchSession(sessionId);
      setMessages(detail.messages);
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

  /** メッセージ送信。 */
  const handleSend = useCallback(async (content: string) => {
    if (!activeSessionId) return;
    setSending(true);
    setError(null);
    try {
      const result = await sendMessage(activeSessionId, content);
      setMessages((prev) => [
        ...prev,
        result.user_message,
        result.character_message,
      ]);
      // セッションタイトルが自動更新された可能性があるのでリロード
      const updated = await fetchSessions();
      setSessions(updated);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [activeSessionId]);

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
            onSend={handleSend}
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
