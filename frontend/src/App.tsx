/**
 * アプリルートコンポーネント。
 * セッション管理・モデル取得・メッセージ送受信のロジックを担当する。
 */
import { useEffect, useState, useCallback, useRef, useMemo } from "react";
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
  fetchDrifts,
  fetchCharacters,
  updateSessionTitle,
} from "./api";
import type { Model, Session, ChatMessage, StreamEvent, GroupStreamEvent, Drift, Character } from "./api";
import { charNameOf } from "./api";
import Sidebar from "./components/Sidebar";
import ChatView from "./components/ChatView";
import GroupChatView from "./components/GroupChatView";
import DriftBadge from "./components/DriftBadge";
import ExportDialog from "./components/ExportDialog";

/** アプリ全体のルートコンポーネント。 */
export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [characters, setCharacters] = useState<Character[]>([]);
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
  /** アクティブセッションのSELF_DRIFT一覧。 */
  const [drifts, setDrifts] = useState<Drift[]>([]);
  /** エクスポートダイアログの開閉状態。 */
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  /** スクロール下方向でヘッダーを隠す。上スクロールまたはセッション未選択時は表示する。 */
  const [headerVisible, setHeaderVisible] = useState(true);

  /** アクティブセッションのキャラクター名を model_id から抽出する。 */
  const activeSession = sessions.find((s) => s.id === activeSessionId);
  /**
   * バブル表示のフォールバック名。character_name未保存の旧メッセージ用。キャラ@プリセット形式で表示する。
   * selectedModel を優先することで、チャット途中でモデル切り替えした直後でも
   * ストリーミング中から正しいモデル名を表示できる。
   */
  const characterName = (selectedModel || activeSession?.model_id) ?? "キャラクター";
  /** アクティブセッションに紐づくキャラクターID（1on1チャット時のDriftBadge用）。 */
  const activeCharacterId = drifts.length > 0 ? drifts[0].character_id : "";
  /** アクティブセッションがグループチャットかどうか。 */
  const isGroupSession = activeSession?.session_type === "group";
  /**
   * キャラクター名→IDのマップ。アバター画像URL生成用。
   * characters が変わるたびに再計算するため useMemo でメモ化する。
   */
  const characterIdMap = useMemo(
    () => Object.fromEntries(characters.map((c) => [c.name, c.id])),
    [characters],
  );
  /** グループチャット参加者情報（char_name・preset_name）。 */
  const groupParticipantEntries = (() => {
    if (!isGroupSession || !activeSession?.group_config) return [];
    try {
      const cfg = JSON.parse(activeSession.group_config);
      return cfg.participants as Array<{ char_name: string; preset_name: string }>;
    } catch {
      return [];
    }
  })();
  /** グループチャット参加者名リスト（色割り当て用）。 */
  const groupParticipantNames = groupParticipantEntries.map((p) => p.char_name);
  /** グループチャット参加者のキャラクター名+ID+プリセット名リスト（ヘッダのDriftBadge用）。 */
  const groupParticipants = groupParticipantEntries.map(({ char_name, preset_name }) => ({
    charName: char_name,
    presetName: preset_name,
    characterId: characters.find((c) => c.name === char_name)?.id ?? "",
  }));
  /**
   * セッション切り替え競合防止用 ref。
   * _doStream / _doGroupStream の非同期処理がセッション切り替え後に state を汚染しないよう、
   * 完了時点でのアクティブセッション ID と比較するために使用する。
   */
  const activeSessionIdRef = useRef<string | null>(activeSessionId);
  activeSessionIdRef.current = activeSessionId;
  /** グループチャット応答待機中・ストリーミング中のキャラクター名（null = なし）。 */
  const [groupWaitingCharacter, setGroupWaitingCharacter] = useState<string | null>(null);
  /** グループチャットストリーミング中の応答テキスト。 */
  const [groupStreamingContent, setGroupStreamingContent] = useState<string | null>(null);
  /** グループチャットストリーミング中の思考ブロック・想起記憶テキスト。 */
  const [groupStreamingReasoning, setGroupStreamingReasoning] = useState<string | null>(null);
  /** グループチャットメッセージIDに紐付いた reasoning テキスト。 */
  const [groupReasoningMap, setGroupReasoningMap] = useState<Record<string, string>>({});
  /** グループチャットがユーザターン待ち状態かどうか。スキップボタンの表示制御に使用する。 */
  const [isGroupUserTurn, setIsGroupUserTurn] = useState(false);

  /** 初期データ取得。URL ハッシュに対応するセッションがあれば自動選択する。 */
  useEffect(() => {
    Promise.all([fetchModels(), fetchSessions(), fetchUserName(), fetchCharacters()])
      .then(([m, s, u, c]) => {
        setModels(m);
        if (m.length > 0) setSelectedModel(m[0].id);
        setSessions(s);
        setUserName(u);
        setCharacters(c);
        // URL ハッシュからセッションを復元する
        const hashSessionId = window.location.hash.slice(1);
        if (hashSessionId && s.find((sess) => sess.id === hashSessionId)) {
          handleSelectSession(hashSessionId);
        }
      })
      .catch((e) => setError(String(e)));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** アクティブセッションのSELF_DRIFT一覧を再取得する。 */
  const refreshDrifts = useCallback(async (sessionId: string) => {
    try {
      const d = await fetchDrifts(sessionId);
      setDrifts(d);
    } catch {
      // drift取得失敗は無視する
    }
  }, []);

  /** セッション選択時にメッセージ一覧を取得し、reasoningMap を復元する。 */
  const handleSelectSession = useCallback(async (sessionId: string) => {
    setActiveSessionId(sessionId);
    setHeaderVisible(true);
    setDrifts([]);
    setError(null);
    // ストリーミング中だった場合の状態をリセットする
    setSending(false);
    setStreamingContent(null);
    setStreamingReasoning(null);
    setGroupWaitingCharacter(null);
    setGroupStreamingContent(null);
    setGroupStreamingReasoning(null);
    setIsGroupUserTurn(false);
    // URL ハッシュを更新して復帰時に同じセッションを開けるようにする
    window.location.hash = sessionId;
    try {
      const [detail] = await Promise.all([
        fetchSession(sessionId),
        fetchDrifts(sessionId).then(setDrifts).catch(() => {}),
      ]);
      // 1on1チャットの場合、そのセッションで最後に使ったモデルをセレクタに反映する
      if (detail.session_type !== "group" && detail.model_id) {
        setSelectedModel(detail.model_id);
      }
      setMessages(detail.messages);
      // グループチャットで最後のメッセージがキャラクター発言ならユーザターン待ち状態を復元する
      if (detail.session_type === "group" && detail.messages.length > 0) {
        const last = detail.messages[detail.messages.length - 1];
        setIsGroupUserTurn(last.role !== "user");
      }
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

  /**
   * 新規チャット作成。
   *
   * @param modelId - "{char_name}@{preset_name}" 形式のモデルID。
   * @param afterglow - Afterglow（感情継続機構）を有効にする場合は true。
   */
  const handleNewChat = useCallback(async (modelId: string, afterglow = false) => {
    setError(null);
    try {
      const session = await createSession(modelId, afterglow);
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setMessages([]);
      // リロード後に同じセッションを復元できるようにハッシュを更新する
      window.location.hash = session.id;
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
      // リロード後に同じセッションを復元できるようにハッシュを更新する
      window.location.hash = session.id;
    } catch (e) {
      setError(String(e));
    }
  }, []);

  /** セッションタイトル変更。 */
  const handleRenameSession = useCallback(async (sessionId: string, newTitle: string) => {
    try {
      const updated = await updateSessionTitle(sessionId, newTitle);
      setSessions((prev) => prev.map((s) => s.id === sessionId ? { ...s, title: updated.title } : s));
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
        window.location.hash = "";
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
   *
   * @param skip - true の場合、ユーザメッセージを保存せず司会へ直接ターンを委譲する（ユーザターンスキップ）。
   */
  const _doGroupStream = useCallback(async (sessionId: string, content: string, imageIds: string[] = [], skip = false) => {
    setError(null);
    setSending(true);
    setIsGroupUserTurn(false);
    setGroupWaitingCharacter(null);

    const optimisticId = `optimistic-${Date.now()}`;
    // スキップ時は楽観的ユーザメッセージを追加しない
    if (!skip) {
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
      for await (const event of streamGroupMessage(sessionId, content, imageIds, skip)) {
        const ev = event as GroupStreamEvent;
        if (ev.type === "user_saved") {
          // optimisticメッセージを確定済みユーザーメッセージで差し替える
          setMessages((prev) => prev.map((m) => m.id === optimisticId ? ev.message : m));
        } else if (ev.type === "character_start") {
          // キャラクター応答開始：スピナーを表示し、前のストリーミング内容をクリアする
          setGroupWaitingCharacter(ev.character);
          setGroupStreamingContent(null);
          setGroupStreamingReasoning(null);
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
          setMessages((prev) => [...prev, ev.message]);
          setGroupStreamingContent(null);
          setGroupStreamingReasoning(null);
          // waitingCharacter は次の character_start か user_turn まで維持する
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
          refreshDrifts(sessionId);
        }
      } catch {
        // 取得失敗は無視する
      }
      setGroupWaitingCharacter(null);
      setGroupStreamingContent(null);
      setGroupStreamingReasoning(null);
      setSending(false);
    }
  }, [refreshDrifts]);

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
    _doGroupStream(activeSessionId, content, imageIds);
  }, [activeSessionId, sending, _doGroupStream]);

  /**
   * グループチャットのユーザターンスキップ。
   * ユーザメッセージを保存せず、司会へ直接ターンを委譲する。
   */
  const handleGroupSkip = useCallback(async () => {
    if (!activeSessionId || sending) return;
    _doGroupStream(activeSessionId, "", [], true);
  }, [activeSessionId, sending, _doGroupStream]);

  /**
   * ストリーミング送信の共通実装。楽観的ユーザメッセージ表示 + SSE受信を行う。
   * handleSend / handleRetry の両方から呼ばれる。
   */
  const _doStream = useCallback(async (sessionId: string, content: string, imageIds: string[] = [], modelId?: string) => {
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
      for await (const event of streamMessage(sessionId, content, imageIds, modelId)) {
        if (event.type === "chunk") {
          setStreamingContent((prev) => (prev ?? "") + event.content);
        } else if (event.type === "clear") {
          // switch_angle 発動: 第1プロバイダーの表示をクリアして第2プロバイダーを待つ
          setStreamingContent("");
          accumulatedReasoning = "";
          setStreamingReasoning(null);
        } else if (event.type === "reasoning") {
          accumulatedReasoning += event.content;
          setStreamingReasoning(accumulatedReasoning);
        } else if (event.type === "done") {
          // セッションが切り替わっていたら state を汚染しない
          if (sessionId !== activeSessionIdRef.current) return;
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
          // SELF_DRIFT が更新されている可能性があるため再取得する
          refreshDrifts(sessionId);
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
   * メッセージ送信。画像がある場合は先にアップロードしてから送信する。
   * グループ・1on1どちらも画像対応済み。
   */
  const handleSend = useCallback(async (content: string, files: File[]) => {
    if (!activeSessionId) return;

    // 画像アップロードはグループ・1on1共通（グループセッションも同じ画像APIを使う）
    let imageIds: string[] = [];
    if (files.length > 0) {
      try {
        const uploaded = await uploadImages(activeSessionId, files);
        imageIds = uploaded.map((u) => u.id);
      } catch (e) {
        setError(String(e));
        return;
      }
    }

    if (isGroupSession) {
      await _doGroupStream(activeSessionId, content, imageIds);
      return;
    }
    setSending(true);
    try {
      await _doStream(activeSessionId, content, imageIds, selectedModel || undefined);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [activeSessionId, isGroupSession, selectedModel, _doGroupStream, _doStream]);

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
      await _doStream(activeSessionId, content, imageIds, selectedModel || undefined);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [activeSessionId, sending, selectedModel, _doStream]);

  return (
    /* h-[100dvh]: モバイルブラウザのアドレスバーを除いた実際の表示領域に合わせる */
    <div className="flex h-[100dvh] overflow-hidden bg-ch-bg text-ch-t1 relative">
      {/* モバイル時: サイドバー背後のオーバーレイ。タップで閉じる。 */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/60 sm:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar
        models={models}
        characters={characters}
        sessions={sessions}
        activeSessionId={activeSessionId}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
        onSelectSession={(id) => { handleSelectSession(id); setSidebarOpen(window.innerWidth >= 640); }}
        onNewChat={handleNewChat}
        onNewGroupChat={handleNewGroupChat}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
      />

      <main className="flex-1 flex flex-col h-full overflow-hidden min-w-0">
        {/* トップバー: ハンバーガートグル + セッション情報。スクロール方向に応じて表示切り替え。 */}
        <div className={`shrink-0 overflow-hidden transition-all duration-300 ease-in-out ${headerVisible ? "max-h-20" : "max-h-0"}`}>
          <div className="flex items-center gap-3 px-3 py-2 bg-ch-bg/95 backdrop-blur-sm" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <button
              onClick={() => setSidebarOpen((o) => !o)}
              className="text-ch-t3 p-1.5 rounded hover:text-ch-t1 transition-colors shrink-0"
              aria-label="サイドバーを開閉"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={18} height={18}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
              </svg>
            </button>

            {/* セッション種別に応じてヘッダ内容を切り替える */}
            {activeSessionId && isGroupSession ? (
              <div className="flex items-center gap-3 flex-wrap min-w-0">
                {groupParticipants.map(({ charName, presetName, characterId }) => {
                  const charDrifts = drifts.filter((d) => d.character_id === characterId);
                  return (
                    <div key={charName} className="flex items-center gap-1.5 shrink-0">
                      <span className="text-ch-t1 text-sm font-medium">{charName}</span>
                      <span className="text-ch-t3 text-xs">@{presetName}</span>
                      {charDrifts.length > 0 && characterId && (
                        <DriftBadge
                          drifts={charDrifts}
                          sessionId={activeSessionId}
                          characterId={characterId}
                          onDriftsChange={() => refreshDrifts(activeSessionId)}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            ) : activeSessionId && !isGroupSession ? (
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-ch-t1 text-sm font-medium truncate">{activeSession?.model_id ?? characterName}</span>
                {activeCharacterId && (
                  <DriftBadge
                    drifts={drifts}
                    sessionId={activeSessionId}
                    characterId={activeCharacterId}
                    onDriftsChange={() => refreshDrifts(activeSessionId)}
                  />
                )}
              </div>
            ) : !sidebarOpen ? (
              <span className="text-ch-t2 text-sm tracking-widest uppercase" style={{ letterSpacing: "0.2em", fontSize: "0.7rem" }}>Chotgor</span>
            ) : null}

            {/* エクスポートボタン */}
            {activeSessionId && messages.length > 0 && (
              <button
                onClick={() => setExportDialogOpen(true)}
                title="会話をエクスポート"
                className="ml-auto text-ch-t3 hover:text-ch-t2 p-1.5 rounded transition-colors"
                aria-label="会話をエクスポート"
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={16} height={16}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* エラーバナー */}
        {error && (
          <div className="bg-red-950/50 text-red-300 text-xs px-4 py-2 flex justify-between items-center shrink-0" style={{ borderBottom: "1px solid rgba(220,60,60,0.15)" }}>
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-200 ml-4">✕</button>
          </div>
        )}

        {activeSessionId && isGroupSession ? (
          <GroupChatView
            sessionId={activeSessionId}
            messages={messages}
            participantNames={groupParticipantNames}
            userName={userName}
            sending={sending}
            waitingCharacter={groupWaitingCharacter}
            streamingContent={groupStreamingContent}
            streamingReasoning={groupStreamingReasoning}
            reasoningMap={groupReasoningMap}
            onSend={handleSend}
            onRetry={handleGroupRetry}
            onHeaderVisibilityChange={setHeaderVisible}
            characterIdMap={characterIdMap}
            isUserTurn={isGroupUserTurn}
            onSkip={handleGroupSkip}
          />
        ) : activeSessionId ? (
          <ChatView
            sessionId={activeSessionId}
            messages={messages}
            characterName={characterName}
            userName={userName}
            sending={sending}
            streamingContent={streamingContent}
            streamingReasoning={streamingReasoning}
            reasoningMap={reasoningMap}
            onSend={handleSend}
            onRetry={handleRetry}
            onHeaderVisibilityChange={setHeaderVisible}
            characterIdMap={characterIdMap}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center text-ch-t3 text-sm px-4 text-center">
            <p>左のサイドバーからチャットを選択するか、新規チャットを作成してください</p>
          </div>
        )}
      </main>

      {/* エクスポートダイアログ */}
      {exportDialogOpen && (
        <ExportDialog
          messages={messages}
          userName={userName}
          reasoningMap={isGroupSession ? groupReasoningMap : reasoningMap}
          sessionTitle={activeSession?.title}
          onClose={() => setExportDialogOpen(false)}
        />
      )}
    </div>
  );
}
