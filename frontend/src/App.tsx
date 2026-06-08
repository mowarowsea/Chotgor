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
  fetchCharacters,
  updateSessionTitle,
  fetchScenarioSessions,
  fetchScenarioPresets,
} from "./api";
import type {
  Model,
  Session,
  ChatMessage,
  StreamEvent,
  Character,
  ScenarioSession,
  ScenarioPreset,
  ScenarioTurn,
} from "./api";
import { charNameOf } from "./api";
import Sidebar from "./components/Sidebar";
import type { AnySession } from "./components/Sidebar";
import ChatView from "./components/ChatView";
import GroupChatView from "./components/GroupChatView";
import ScenarioChatView from "./components/ScenarioChatView";
import ExportDialog from "./components/ExportDialog";
import { CharacterAvatar, CharacterImageProvider } from "./components/ChatBubbles";
import CharPresetMenu from "./components/CharPresetMenu";
import ScenarioSettingsModal from "./components/ScenarioSettingsModal";
import SynopsisCreateModal from "./components/SynopsisCreateModal";
import { useTheme } from "./hooks/useTheme";
import { useGroupChat } from "./hooks/useGroupChat";
import { useScenarioChat } from "./hooks/useScenarioChat";

/**
 * ScenarioTurns を ExportDialog が期待する ChatMessage 形式に変換する。
 *
 * - user 発話: role="user"、speaker_name は無視（template 側で userName を使う）
 * - GM 系（narrator / npc / character）: role="character"、character_name に
 *   speaker_name を入れる（narrator は "Narrator" にする）
 *
 * intro 由来のターンや auto_advance ターンも、speaker_type に応じて
 * 上記の単純なマッピングで出力される。
 */
function scenarioTurnsToExportMessages(
  turns: ScenarioTurn[],
  sessionId: string,
): ChatMessage[] {
  return turns.map<ChatMessage>((t) => {
    if (t.speaker_type === "user") {
      return {
        id: t.id,
        session_id: sessionId,
        role: "user",
        content: t.content,
        created_at: t.created_at,
      };
    }
    const charName = t.speaker_type === "narrator" ? "Narrator" : t.speaker_name;
    return {
      id: t.id,
      session_id: sessionId,
      role: "character",
      content: t.content,
      character_name: charName,
      created_at: t.created_at,
    };
  });
}

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
  /** エクスポートダイアログの開閉状態。 */
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  /** ライト/ダークテーマの状態と切り替え関数。 */
  const { dark, toggle: toggleTheme } = useTheme();
  /** ヘッダーのモデル切り替えメニューの開閉状態。 */
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  /**
   * 浮遊ヘッダーの表示状態。
   * 上スクロール（過去ログ閲覧）で隠れ、下スクロール・最下部・1画面に収まる場合は表示する。
   */
  const [headerVisible, setHeaderVisible] = useState(true);

  /** アクティブセッションのキャラクター名を model_id から抽出する。 */
  const activeSession = sessions.find((s) => s.id === activeSessionId);
  /**
   * バブル表示のフォールバック名。character_name未保存の旧メッセージ用。キャラ@プリセット形式で表示する。
   * selectedModel を優先することで、チャット途中でモデル切り替えした直後でも
   * ストリーミング中から正しいモデル名を表示できる。
   */
  const characterName = (selectedModel || activeSession?.model_id) ?? "キャラクター";
  /** アクティブセッションがグループチャットかどうか。 */
  const isGroupSession = activeSession?.session_type === "group";
  /**
   * キャラクター名→アバター画像URLのリゾルバ。
   * CharacterImageProvider 経由でアプリ全体の CharacterAvatar が参照する。
   * characters が変わるたびに再計算するため useMemo でメモ化する。
   */
  const resolveCharImage = useMemo(() => {
    const idByName = new Map(characters.map((c) => [c.name, c.id]));
    return (name: string): string | undefined => {
      const id = idByName.get(name);
      return id ? `/api/characters/${id}/image` : undefined;
    };
  }, [characters]);
  /** グループチャット参加者情報（char_name・preset_name）。 */
  const groupParticipantEntries = (() => {
    if (!isGroupSession || !activeSession?.group_config) return [];
    try {
      const cfg = JSON.parse(activeSession.group_config);
      return cfg.participants as Array<{ char_name: string; preset_id: string; preset_name: string }>;
    } catch {
      return [];
    }
  })();
  /** グループチャット参加者名リスト（色割り当て用）。 */
  const groupParticipantNames = groupParticipantEntries.map((p) => p.char_name);
  /**
   * グループメッセージから char_name → preset_name のフォールバックマップを作る。
   * group_config に preset_name を持たない旧セッション向けの救済。
   * 各キャラクター発言メッセージは preset_name を持つため、それを参照する。
   */
  const groupPresetFallback = useMemo(() => {
    const m = new Map<string, string>();
    for (const msg of messages) {
      if (msg.role !== "user" && msg.character_name && msg.preset_name) {
        m.set(msg.character_name, msg.preset_name);
      }
    }
    return m;
  }, [messages]);
  /** グループチャット参加者のキャラクター名+プリセット名リスト（ヘッダのアバター・名前表示用）。 */
  const groupParticipants = groupParticipantEntries.map(({ char_name, preset_name }) => ({
    charName: char_name,
    // group_config の preset_name（新セッション）→ 旧セッションはメッセージから補完。
    presetName: preset_name || groupPresetFallback.get(char_name) || "",
  }));
  /**
   * セッション切り替え競合防止用 ref。
   * _doStream / _doGroupStream の非同期処理がセッション切り替え後に state を汚染しないよう、
   * 完了時点でのアクティブセッション ID と比較するために使用する。
   */
  const activeSessionIdRef = useRef<string | null>(activeSessionId);
  activeSessionIdRef.current = activeSessionId;
  /** char_msg_id → log_message_id（8桁hex）のマッピング。バブルのログ折りたたみに使用する。 */
  const [msgLogIds, setMsgLogIds] = useState<Record<string, string>>({});
  /**
   * メッセージID（1on1/グループの char_msg_id、シナリオの turn_id）
   * → モデルへリクエストしてから応答完了までの経過時間（ミリ秒）のマッピング。
   * 現セッションのストリーミング分のみ保持し、ページリロードで消える。
   */
  const [elapsedMap, setElapsedMap] = useState<Record<string, number>>({});

  /**
   * グループチャットの state・ストリーミング送受信ハンドラ群。
   * 共有 state（messages / sessions / sending 等）の setter を渡し、group 専用 state は
   * フック内に閉じ込める。返り値の setter（reasoning / userTurn）はセッション復元に使う。
   */
  const {
    groupWaitingCharacter,
    groupStreamingContent,
    groupStreamingReasoning,
    groupReasoningMap,
    isGroupUserTurn,
    groupDirectorErrored,
    setGroupReasoningMap,
    setIsGroupUserTurn,
    resetStreamingState: resetGroupStreamingState,
    doGroupStream,
    handleGroupRetry,
    handleGroupSkip,
    handleGroupRetryDirector,
    handleGroupRequestCharacter,
  } = useGroupChat({
    activeSessionId,
    sending,
    activeSessionIdRef,
    setMessages,
    setSessions,
    setError,
    setSending,
    setElapsedMap,
  });

  /**
   * シナリオプレイの state・ストリーミング送受信ハンドラ群。
   * 共有 state（activeSessionId / sending / elapsedMap 等）の setter を渡し、シナリオ専用
   * state はフック内に閉じ込める。返り値の setter・ref はセッション選択/復元で使う。
   */
  const {
    scenarioSessions,
    scenarioSessionsRef,
    activeScenarioSession,
    activeScenarioTemplate,
    scenarioPresets,
    scenarioPresetName,
    scenarioNpcs,
    scenarioTurns,
    scenarioPending,
    scenarioSynopsis,
    synopsisGenerating,
    synopsisModalOpen,
    scenarioSettingsTab,
    synopsisBar,
    setScenarioSessions,
    setScenarioPresets,
    setScenarioSettingsTab,
    resetScenarioState,
    loadScenarioSession,
    deleteScenario,
    handleStartScenario,
    handleScenarioPresetChange,
    handleScenarioSend,
    handleScenarioRegenerate,
    handleScenarioDiscard,
    handleScenarioEditUserTurn,
    handleSynopsisChange,
    handleSynopsisCreate,
    handleOpenSynopsisCreate,
    handleCancelSynopsisCreate,
  } = useScenarioChat({
    activeSessionId,
    setActiveSessionId,
    activeSessionIdRef,
    setSending,
    setError,
    setElapsedMap,
  });

  /** Sidebar に渡す統合セッションリスト（更新日時降順）。 */
  const combinedSessions: AnySession[] = useMemo(() => {
    const all: AnySession[] = [...sessions, ...scenarioSessions];
    return all.sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1));
  }, [sessions, scenarioSessions]);

  /** 現在選択中セッションがシナリオかどうか。 */
  const isScenarioSession = combinedSessions.find(
    (s) => s.id === activeSessionId,
  )?.session_type === "scenario";

  /** 初期データ取得。URL ハッシュに対応するセッションがあれば自動選択する。 */
  useEffect(() => {
    Promise.all([
      fetchModels(),
      fetchSessions(),
      fetchUserName(),
      fetchCharacters(),
      fetchScenarioSessions().catch(() => [] as ScenarioSession[]),
      fetchScenarioPresets().catch(() => [] as ScenarioPreset[]),
    ])
      .then(([m, s, u, c, sc, sp]) => {
        setModels(m);
        if (m.length > 0) setSelectedModel(m[0].id);
        setSessions(s);
        setUserName(u);
        setCharacters(c);
        setScenarioSessions(sc);
        setScenarioPresets(sp);
        // ref も即座に更新する。次行の handleSelectSession() がこの ref を
        // 読みに来るため、setState の非同期反映を待たずに同期反映が必要。
        scenarioSessionsRef.current = sc;
        // URL ハッシュからセッションを復元する（chat / scenario の両方をチェック）
        const hashSessionId = window.location.hash.slice(1);
        if (hashSessionId) {
          if (s.find((sess) => sess.id === hashSessionId)) {
            handleSelectSession(hashSessionId);
          } else if (sc.find((sess) => sess.id === hashSessionId)) {
            handleSelectSession(hashSessionId);
          }
        }
      })
      .catch((e) => setError(String(e)));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** セッション選択時にメッセージ一覧を取得し、reasoningMap を復元する。 */
  const handleSelectSession = useCallback(async (sessionId: string) => {
    setActiveSessionId(sessionId);
    setHeaderVisible(true);
    setError(null);
    // ストリーミング中だった場合の状態をリセットする
    setSending(false);
    setStreamingContent(null);
    setStreamingReasoning(null);
    resetGroupStreamingState();
    // シナリオ関連の状態もリセット
    resetScenarioState();
    // URL ハッシュを更新して復帰時に同じセッションを開けるようにする
    window.location.hash = sessionId;

    // セッション種別を一覧から判別する（scenarios と sessions の両方を見る）。
    // ref 経由で参照するのは、初回マウント時のハッシュ復元で setScenarioSessions 直後に
    // この関数が呼ばれた際、state closure はまだ空 [] のままで scenario と判定できないため。
    const isScenario = scenarioSessionsRef.current.some((s) => s.id === sessionId);
    if (isScenario) {
      await loadScenarioSession(sessionId);
      return;
    }

    try {
      const [detail] = await Promise.all([
        fetchSession(sessionId),
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
      // DBに保存された reasoning と log_message_id をメッセージIDに紐付けて復元する
      const restored: Record<string, string> = {};
      const restoredLogIds: Record<string, string> = {};
      for (const msg of detail.messages) {
        if (msg.reasoning) {
          restored[msg.id] = msg.reasoning;
        }
        if (msg.log_message_id) {
          restoredLogIds[msg.id] = msg.log_message_id;
        }
      }
      setReasoningMap(restored);
      setGroupReasoningMap(restored);
      setMsgLogIds(restoredLogIds);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  /**
   * 新規チャット作成。
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
  }, []);

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

  /** セッション削除。session_type を判別して適切な API を呼ぶ。 */
  const handleDeleteSession = useCallback(async (sessionId: string) => {
    setError(null);
    const isScenario = scenarioSessions.some((s) => s.id === sessionId);
    try {
      if (isScenario) {
        await deleteScenario(sessionId);
      } else {
        await deleteSession(sessionId);
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      }
      if (activeSessionId === sessionId) {
        window.location.hash = "";
        setActiveSessionId(null);
        setMessages([]);
        resetScenarioState();
      }
    } catch (e) {
      setError(String(e));
    }
  }, [activeSessionId, scenarioSessions, deleteScenario, resetScenarioState]);

  /**
   * ストリーミング送信の共通実装。楽観的ユーザメッセージ表示 + SSE受信を行う。
   * handleSend / handleRetry の両方から呼ばれる。
   */
  const _doStream = useCallback(async (sessionId: string, content: string, imageIds: string[] = [], modelId?: string) => {
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
    try {
      for await (const event of streamMessage(sessionId, content, imageIds, modelId)) {
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
          // セッションが切り替わっていたら state を汚染しない
          if (sessionId !== activeSessionIdRef.current) return;
          if (accumulatedReasoning) {
            setReasoningMap((prev) => ({
              ...prev,
              [event.character_message.id]: accumulatedReasoning,
            }));
          }
          // デバッグログIDをバブルと紐付ける
          if (event.log_message_id) {
            setMsgLogIds((prev) => ({
              ...prev,
              [event.character_message.id]: event.log_message_id!,
            }));
          }
          // 経過時間を記録する（ストリーム送信〜done受信）
          setElapsedMap((prev) => ({
            ...prev,
            [event.character_message.id]: performance.now() - streamStartedAt,
          }));
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
      await doGroupStream(activeSessionId, content, imageIds);
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
  }, [activeSessionId, isGroupSession, selectedModel, doGroupStream, _doStream]);

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
    /* CharacterImageProvider: アバター画像リゾルバをアプリ全体へ供給する。 */
    <CharacterImageProvider resolve={resolveCharImage}>
    {/* h-[100dvh]: モバイルブラウザのアドレスバーを除いた実際の表示領域に合わせる */}
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
        sessions={combinedSessions}
        activeSessionId={activeSessionId}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
        onSelectSession={(id) => { handleSelectSession(id); setSidebarOpen(window.innerWidth >= 640); }}
        onNewChat={handleNewChat}
        onNewGroupChat={handleNewGroupChat}
        onStartScenario={handleStartScenario}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
      />

      <main className="relative flex-1 flex flex-col h-full overflow-hidden min-w-0">
        {/* 浮遊ヘッダー: 常時表示・背景は透過。コンテンツはこの下をスクロールする。
            バー自体は pointer-events-none で、操作可能なピルだけ pointer-events-auto。 */}
        <div className="absolute top-0 left-0 right-0 z-20 pointer-events-none">
          <div
            className={`flex items-center gap-2 px-3 py-2.5 transition-all duration-300 ${
              headerVisible ? "opacity-100 translate-y-0 visible" : "opacity-0 -translate-y-3 invisible"
            }`}
          >
            {/* サイドバートグル（ピルボタン） */}
            <button
              onClick={() => setSidebarOpen((o) => !o)}
              className="pointer-events-auto shrink-0 flex items-center justify-center rounded-lg bg-ch-bg text-ch-t3 hover:text-ch-t1 transition-colors"
              style={{ border: "1px solid var(--ch-sep2)", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", padding: "6px 8px" }}
              aria-label="サイドバーを開閉"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={16} height={16}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
              </svg>
            </button>

            {/* タイトルチップ（角丸ピル）。セッション種別に応じて内容を切り替える。 */}
            {activeSessionId && isScenarioSession && activeScenarioSession ? (
              <div
                className="pointer-events-auto relative flex items-center gap-2 rounded-full bg-ch-bg min-w-0 max-w-[60%]"
                style={{ border: "1px solid var(--ch-sep2)", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", padding: "5px 12px 5px 6px" }}
              >
                {/* シナリオタイトル + GM プリセット名のチップ。クリックでシナリオ設定モーダルを開く。
                    モーダル内でシナリオ用モデル・あらすじ用モデル・あらすじ閲覧/編集を切替可能。 */}
                <button
                  onClick={() => setScenarioSettingsTab("model")}
                  className="relative flex items-center gap-2 min-w-0"
                  title="シナリオ設定（モデル / あらすじ）"
                >
                  <span className="shrink-0 w-6 h-6 rounded-full bg-ch-s1 flex items-center justify-center text-ch-t2 text-xs">✦</span>
                  <div className="min-w-0 text-left">
                    <div className="text-ch-t1 text-[13px] font-semibold truncate leading-tight">{activeScenarioSession.title}</div>
                    <div className="text-ch-t3 text-[10px] font-mono leading-tight truncate">
                      {scenarioPresetName ? `@${scenarioPresetName}` : "scenario"}
                      <span className="text-ch-t4 ml-1">▾</span>
                    </div>
                  </div>
                </button>
              </div>
            ) : activeSessionId && isGroupSession ? (
              <div
                className="pointer-events-auto flex items-center gap-2 rounded-full bg-ch-bg min-w-0 max-w-[64%]"
                style={{ border: "1px solid var(--ch-sep2)", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", padding: "5px 12px 5px 6px" }}
              >
                {/* 参加者アバターを重ねて表示 */}
                <div className="flex shrink-0">
                  {groupParticipants.map(({ charName }, i) => (
                    <div key={charName} style={{ marginLeft: i ? -8 : 0, zIndex: 10 - i }}>
                      <CharacterAvatar characterName={charName} size={24} />
                    </div>
                  ))}
                </div>
                <div className="flex items-center gap-2.5 flex-wrap min-w-0">
                  {groupParticipants.map(({ charName, presetName }) => (
                    <div key={charName} className="flex items-center gap-1 shrink-0">
                      <span className="text-ch-t1 text-[13px] font-semibold">{charName}</span>
                      {presetName && (
                        <span className="text-ch-t3 text-[10px] font-mono">@{presetName}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : activeSessionId && !isGroupSession ? (
              <div
                className="pointer-events-auto relative flex items-center gap-1.5 rounded-full bg-ch-bg min-w-0 max-w-[60%]"
                style={{ border: "1px solid var(--ch-sep2)", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", padding: "5px 12px 5px 6px" }}
              >
                {/* モデル名チップ。クリックでモデル切り替えメニューを開く。 */}
                <button
                  onClick={() => setModelMenuOpen((o) => !o)}
                  className="flex items-center gap-1.5 min-w-0"
                  title="モデルを切り替え"
                >
                  <CharacterAvatar characterName={charNameOf(selectedModel || activeSession?.model_id || characterName)} size={24} />
                  <span className="text-ch-t1 text-[13px] font-semibold truncate">{selectedModel || activeSession?.model_id || characterName}</span>
                  <span className="text-ch-t3 text-[9px] shrink-0">▾</span>
                </button>
                {/* モデル切り替えメニュー（キャラ/プリセット2段選択） */}
                {modelMenuOpen && (
                  <CharPresetMenu
                    models={models}
                    currentModelId={selectedModel || activeSession?.model_id || ""}
                    onApply={setSelectedModel}
                    onClose={() => setModelMenuOpen(false)}
                  />
                )}
              </div>
            ) : !sidebarOpen ? (
              <span className="pointer-events-auto text-ch-t1 font-bold text-[15px]" style={{ letterSpacing: "-0.02em" }}>Chotgor</span>
            ) : null}

            {/* 右側ボタン群: テーマ切り替え + エクスポート（ピルボタン）。
                シナリオの「あらすじ」ボタンはモデルチップ側のモーダルに統合済み。 */}
            <div className="pointer-events-auto ml-auto flex items-center gap-1.5">
              <button
                onClick={toggleTheme}
                title={dark ? "ライトテーマに切り替え" : "ダークテーマに切り替え"}
                className="flex items-center justify-center rounded-lg bg-ch-bg text-ch-t3 hover:text-ch-t1 transition-colors"
                style={{ border: "1px solid var(--ch-sep2)", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", padding: "6px 8px" }}
                aria-label="テーマを切り替え"
              >
                {dark ? (
                  /* 太陽アイコン（ライトへ） */
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={16} height={16}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.25m6.364.386-1.591 1.591M21 12h-2.25m-.386 6.364-1.591-1.591M12 18.75V21m-4.773-4.227-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0Z" />
                  </svg>
                ) : (
                  /* 月アイコン（ダークへ） */
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={16} height={16}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21.752 15.002A9.718 9.718 0 0 1 18 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 0 0 3 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 0 0 9.002-5.998Z" />
                  </svg>
                )}
              </button>
              {activeSessionId &&
                ((isScenarioSession && scenarioTurns.length > 0) ||
                  (!isScenarioSession && messages.length > 0)) && (
                <button
                  onClick={() => setExportDialogOpen(true)}
                  title="会話をエクスポート"
                  className="flex items-center justify-center rounded-lg bg-ch-bg text-ch-t3 hover:text-ch-t1 transition-colors"
                  style={{ border: "1px solid var(--ch-sep2)", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", padding: "6px 8px" }}
                  aria-label="会話をエクスポート"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={16} height={16}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
                  </svg>
                </button>
              )}
            </div>
          </div>

          {/* エラーバナー（浮遊ヘッダー内・ヘッダーバーの下に表示） */}
          {error && (
            <div className="pointer-events-auto mx-3 mb-1 rounded-lg bg-red-500/10 text-red-600 dark:text-red-300 text-xs px-3 py-2 flex justify-between items-center" style={{ border: "1px solid rgba(220,60,60,0.25)" }}>
              <span>{error}</span>
              <button onClick={() => setError(null)} className="text-red-500 hover:text-red-400 ml-4">✕</button>
            </div>
          )}
        </div>

        {activeSessionId && isScenarioSession && activeScenarioSession ? (
          <ScenarioChatView
            session={activeScenarioSession}
            scenario={activeScenarioTemplate}
            npcs={scenarioNpcs}
            turns={scenarioTurns}
            sending={sending}
            pendingBubbles={scenarioPending}
            onSend={handleScenarioSend}
            onEditUserTurn={handleScenarioEditUserTurn}
            onRegenerate={handleScenarioRegenerate}
            onDiscard={handleScenarioDiscard}
            onHeaderVisibilityChange={setHeaderVisible}
            elapsedMap={elapsedMap}
            synopsisBar={synopsisBar}
            synopsisGenerating={synopsisGenerating}
            onOpenSynopsisCreate={handleOpenSynopsisCreate}
            synopsisLastTurnIndex={scenarioSynopsis?.last_turn_index ?? -1}
          />
        ) : activeSessionId && isGroupSession ? (
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
            isUserTurn={isGroupUserTurn}
            onSkip={handleGroupSkip}
            directorErrored={groupDirectorErrored}
            onRetryDirector={handleGroupRetryDirector}
            onRequestCharacter={handleGroupRequestCharacter}
            elapsedMap={elapsedMap}
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
            msgLogIds={msgLogIds}
            elapsedMap={elapsedMap}
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
          messages={
            isScenarioSession
              ? scenarioTurnsToExportMessages(scenarioTurns, activeScenarioSession?.id ?? "")
              : messages
          }
          userName={
            isScenarioSession
              ? (activeScenarioTemplate?.user_alias ?? userName)
              : userName
          }
          reasoningMap={isGroupSession ? groupReasoningMap : reasoningMap}
          sessionTitle={
            isScenarioSession
              ? activeScenarioSession?.title
              : activeSession?.title
          }
          onClose={() => setExportDialogOpen(false)}
        />
      )}

      {/* シナリオセッション設定モーダル（モデル切替 + あらすじ閲覧/編集）。
          左上モデルチップから開く。タブで「モデル」「あらすじ」を切り替える。
          synopsis タブを開いた直後にバッジを下げる（既読扱い）。 */}
      {scenarioSettingsTab !== null &&
        isScenarioSession &&
        activeScenarioSession && (
          <ScenarioSettingsModal
            presets={scenarioPresets}
            currentGmPresetId={activeScenarioSession.gm_preset_id}
            onApplyGmPreset={handleScenarioPresetChange}
            synopsis={scenarioSynopsis}
            onSynopsisChange={handleSynopsisChange}
            onOpenSynopsisCreate={handleOpenSynopsisCreate}
            disabled={sending}
            initialTab={scenarioSettingsTab}
            onClose={() => setScenarioSettingsTab(null)}
          />
        )}

      {/* あらすじ作成モーダル。閾値到達時の自動表示、または設定モーダル/バーから手動表示。
          「作成」で裏蒸留、「キャンセル」でバー表示へ降格する。 */}
      {synopsisModalOpen &&
        isScenarioSession &&
        activeScenarioSession && (
          <SynopsisCreateModal
            presets={scenarioPresets}
            currentSynopsisPresetId={activeScenarioSession.synopsis_preset_id}
            onCreate={handleSynopsisCreate}
            onCancel={handleCancelSynopsisCreate}
          />
        )}
    </div>
    </CharacterImageProvider>
  );
}
