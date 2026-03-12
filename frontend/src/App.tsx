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
  uploadImages,
  streamMessage,
  fetchUserName,
  createGroupSession,
  streamGroupMessage,
} from "./api";
import type { Model, Session, ChatMessage, StreamEvent, GroupStreamEvent } from "./api";
import Sidebar from "./components/Sidebar";
import ChatView from "./components/ChatView";
import GroupChatView from "./components/GroupChatView";

/** アプリ全体のルートコンポーネント。 */
export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  /** サイドバーの開閉状態。デスクトップはデフォルト開、モバイルはデフォルト閉。 */
  const [sidebarOpen, setSidebarOpen] = useState(typeof window !== "undefined" ? window.innerWidth >= 640 : true);
  /** ストリーミング中の思考ブロック・想起記憶テキスト（null = なし） */
  const [streamingReasoning, setStreamingReasoning] = useState<string | null>(null);
  /** 完了済みメッセージIDに紐付いた reasoning テキスト。ページリロードまで保持する。 */
  const [reasoningMap, setReasoningMap] = useState<Record<string, string>>({});
  const [userName, setUserName] = useState("ユーザ");
  const [error, setError] = useState<string | null>(null);

  /** アクティブセッションのキャラクター名を model_id から抽出する。 */
  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const characterName = activeSession?.model_id.split("@")[0] ?? "キャラクター";
  /** アクティブセッションがグループチャットかどうか。 */
  const isGroupSession = activeSession?.session_type === "group";
  /** グループチャット参加者名リスト。 */
  const groupParticipantNames = (() => {
    if (!isGroupSession || !activeSession?.group_config) return [];
    try {
      const cfg = JSON.parse(activeSession.group_config);
      return (cfg.participants as Array<{ char_name: string }>).map((p) => p.char_name);
    } catch {
      return [];
    }
  })();
  /** グループチャット応答待機中のキャラクター名（null = 待機なし）。 */
  const [groupWaitingCharacter, setGroupWaitingCharacter] = useState<string | null>(null);
  /** グループチャットメッセージIDに紐付いた reasoning テキスト。 */
  const [groupReasoningMap, setGroupReasoningMap] = useState<Record<string, string>>({});

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
      setGroupReasoningMap(restored);
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

  /** 新規グループチャット作成。 */
  const handleNewGroupChat = useCallback(async (
    participants: string[],
    directorModelId: string,
    maxAutoTurns: number,
  ) => {
    setError(null);
    try {
      const session = await createGroupSession(participants, directorModelId, maxAutoTurns, 30);
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
   * グループチャットのメッセージ送信実装。
   * SSE受信中はキャラクター名を waitingCharacter で表示し、
   * 受信完了後に全メッセージをサーバーから再取得して確定する。
   */
  const _doGroupStream = useCallback(async (sessionId: string, content: string) => {
    setError(null);
    setSending(true);
    setGroupWaitingCharacter(null);

    const optimisticId = `optimistic-${Date.now()}`;
    const optimisticUserMsg: ChatMessage = {
      id: optimisticId,
      session_id: sessionId,
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUserMsg]);

    try {
      for await (const event of streamGroupMessage(sessionId, content)) {
        const ev = event as GroupStreamEvent;
        if (ev.type === "user_saved") {
          // optimisticメッセージを確定済みユーザーメッセージで差し替える
          setMessages((prev) => prev.map((m) => m.id === optimisticId ? ev.message : m));
        } else if (ev.type === "speaker_decided" && ev.speakers.length > 0) {
          setGroupWaitingCharacter(ev.speakers[0]);
        } else if (ev.type === "character_message") {
          setGroupWaitingCharacter(null);
          // reasoning を reasoningMap に保存する
          if (ev.message.reasoning) {
            setGroupReasoningMap((prev) => ({ ...prev, [ev.message.id]: ev.message.reasoning! }));
          }
          setMessages((prev) => [...prev, ev.message]);
        } else if (ev.type === "user_turn" || ev.type === "done") {
          setGroupWaitingCharacter(null);
          break;
        } else if (ev.type === "error") {
          setGroupWaitingCharacter(null);
          setError(ev.message);
          break;
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      // サーバーから最新状態を取得して確定する
      try {
        const detail = await fetchSession(sessionId);
        setMessages(detail.messages);
        const updated = await fetchSessions();
        setSessions(updated);
      } catch {
        // 取得失敗は無視する
      }
      setGroupWaitingCharacter(null);
      setSending(false);
    }
  }, []);

  /**
   * グループチャットの編集・再生成共通ハンドラ。
   * fromMessageId 以降をDBから削除し、content でグループストリームを再送する。
   */
  const handleGroupRetry = useCallback(async (fromMessageId: string, content: string) => {
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
    _doGroupStream(activeSessionId, content);
  }, [activeSessionId, sending, _doGroupStream]);

  /**
   * ストリーミング送信の共通実装。楽観的ユーザメッセージ表示 + SSE受信を行う。
   * handleSend / handleRetry の両方から呼ばれる。
   */
  const _doStream = useCallback(async (sessionId: string, content: string, imageIds: string[] = []) => {
    setError(null);
    setStreamingContent("");
    setStreamingReasoning(null);

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
    try {
      for await (const event of streamMessage(sessionId, content, imageIds)) {
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

  /**
   * メッセージ送信。グループセッションと1on1セッションで処理を分岐する。
   * 1on1: 画像があれば先にアップロードしてからストリーミング送信する。
   * グループ: テキストのみ送信（画像非対応）。
   */
  const handleSend = useCallback(async (content: string, files: File[]) => {
    if (!activeSessionId) return;
    if (isGroupSession) {
      await _doGroupStream(activeSessionId, content);
      return;
    }
    setSending(true);
    try {
      let imageIds: string[] = [];
      if (files.length > 0) {
        const uploaded = await uploadImages(activeSessionId, files);
        imageIds = uploaded.map((u) => u.id);
      }
      await _doStream(activeSessionId, content, imageIds);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [activeSessionId, isGroupSession, _doGroupStream, _doStream]);

  /**
   * ユーザメッセージ編集 / キャラクター応答再生成の共通ハンドラ。
   * fromMessageId 以降をDBから削除し、content でストリームを再送する。
   * 再生成の場合は fromMessageId = 直前ユーザメッセージのID、content = そのメッセージ本文。
   * imageIds = 再送する画像IDリスト（再生成時は元メッセージの画像を引き継ぐ）。
   */
  const handleRetry = useCallback(async (fromMessageId: string, content: string, imageIds: string[] = []) => {
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
      await _doStream(activeSessionId, content, imageIds);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [activeSessionId, sending, _doStream]);

  return (
    /* h-[100dvh]: モバイルブラウザのアドレスバーを除いた実際の表示領域に合わせる */
    <div className="flex h-[100dvh] overflow-hidden bg-zinc-950 text-zinc-100 relative">
      {/* モバイル時: サイドバー背後のオーバーレイ。タップで閉じる。 */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/50 sm:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar
        models={models}
        sessions={sessions}
        activeSessionId={activeSessionId}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
        onSelectSession={(id) => { handleSelectSession(id); setSidebarOpen(window.innerWidth >= 640); }}
        onNewChat={handleNewChat}
        onNewGroupChat={handleNewGroupChat}
        onDeleteSession={handleDeleteSession}
      />

      <main className="flex-1 flex flex-col h-full overflow-hidden min-w-0">
        {/* トップバー: ハンバーガートグルボタン。サイドバーの開閉を制御する。 */}
        <div className="flex items-center gap-3 px-3 py-2.5 bg-zinc-900 border-b border-zinc-700 shrink-0">
          <button
            onClick={() => setSidebarOpen((o) => !o)}
            className="text-zinc-100 p-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 active:bg-zinc-600 transition-colors"
            aria-label="サイドバーを開閉"
          >
            {/* ハンバーガーアイコン */}
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" width={20} height={20}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
            </svg>
          </button>
          {/* サイドバーが閉じているときはタイトルを表示 */}
          {!sidebarOpen && (
            <span className="text-zinc-100 text-sm font-semibold">Chotgor</span>
          )}
        </div>

        {/* エラーバナー */}
        {error && (
          <div className="bg-red-900/60 text-red-200 text-sm px-4 py-2 flex justify-between items-center shrink-0">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-red-300 hover:text-red-100">✕</button>
          </div>
        )}

        {activeSessionId && isGroupSession ? (
          <GroupChatView
            messages={messages}
            participantNames={groupParticipantNames}
            userName={userName}
            sending={sending}
            waitingCharacter={groupWaitingCharacter}
            reasoningMap={groupReasoningMap}
            onSend={(content) => handleSend(content, [])}
            onRetry={handleGroupRetry}
          />
        ) : activeSessionId ? (
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
          <div className="flex-1 flex items-center justify-center text-zinc-500 text-sm px-4 text-center">
            <p>左のサイドバーからチャットを選択するか、新規チャットを作成してください</p>
          </div>
        )}
      </main>
    </div>
  );
}
