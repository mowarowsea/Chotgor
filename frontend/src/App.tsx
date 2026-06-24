/**
 * アプリルートコンポーネント。
 * セッション管理・モデル取得・メッセージ送受信のロジックを担当する。
 */
import { useEffect, useState, useCallback, useMemo } from "react";
import {
  fetchModels,
  fetchSessions,
  fetchSession,
  deleteSession,
  uploadImages,
  fetchUserName,
  fetchCharacters,
  fetchScenarioSessions,
  fetchScenarioPresets,
  updateFaceToFaceMode,
} from "./api";
import type {
  Model,
  Session,
  ChatMessage,
  Character,
  ScenarioSession,
  ScenarioPreset,
  ScenarioTurn,
} from "./api";
import { charNameOf } from "./api";
import Sidebar from "./components/Sidebar";
import type { AnySession } from "./components/Sidebar";
import ChatView from "./components/ChatView";
import ScenarioChatView from "./components/ScenarioChatView";
import ExportDialog from "./components/ExportDialog";
import { CharacterAvatar, CharacterImageProvider } from "./components/ChatBubbles";
import CharPresetMenu from "./components/CharPresetMenu";
import ScenarioSettingsModal from "./components/ScenarioSettingsModal";
import SynopsisCreateModal from "./components/SynopsisCreateModal";
import { useTheme } from "./hooks/useTheme";
import { useSessions } from "./hooks/useSessions";
import { useChat } from "./hooks/useChat";
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
  const [characters, setCharacters] = useState<Character[]>([]);
  const [sending, setSending] = useState(false);
  /** サイドバーの開閉状態。デスクトップはデフォルト開、モバイルはデフォルト閉。 */
  const [sidebarOpen, setSidebarOpen] = useState(typeof window !== "undefined" ? window.innerWidth >= 640 : true);
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

  /**
   * セッション一覧・中核 state（sessions / activeSessionId / messages）と
   * 自己完結するセッション操作ハンドラ群。
   * useScenarioChat より先に呼び、返す setter・ref をそれらへ渡す。
   */
  const {
    sessions,
    setSessions,
    activeSessionId,
    setActiveSessionId,
    messages,
    setMessages,
    activeSessionIdRef,
    handleNewChat,
    handleRenameSession,
  } = useSessions({ setError, setSelectedModel });

  /** アクティブセッションのキャラクター名を model_id から抽出する。 */
  const activeSession = sessions.find((s) => s.id === activeSessionId);
  /**
   * バブル表示のフォールバック名。character_name未保存の旧メッセージ用。キャラ@プリセット形式で表示する。
   * selectedModel を優先することで、チャット途中でモデル切り替えした直後でも
   * ストリーミング中から正しいモデル名を表示できる。
   */
  const characterName = (selectedModel || activeSession?.model_id) ?? "キャラクター";
  /**
   * 1on1チャットの「現在のキャラクター」とその対面モード状態を導出する。
   *
   * - 対面モードはキャラスコープ（characters.face_to_face_mode）。
   * - 別セッションでの切替はリアルタイム同期しない方針（次回オープン時に反映）。
   * - 切替時は characters state を楽観更新し、API を叩いて確定する。
   */
  const activeCharNameOnly = charNameOf(selectedModel || activeSession?.model_id || characterName);
  const activeCharacter = characters.find((c) => c.name === activeCharNameOnly);
  const faceToFaceMode = !!activeCharacter?.face_to_face_mode;
  const faceToFaceBgUrl = activeCharacter && activeCharacter.has_face_to_face_bg_image
    ? `/api/characters/${activeCharacter.id}/face_to_face_bg_image`
    : null;
  /** 対面モード切替: 楽観更新 → API。失敗時は state を巻き戻してエラー表示。 */
  const handleToggleFaceToFace = useCallback(async (enabled: boolean) => {
    if (!activeCharacter) return;
    const prev = activeCharacter.face_to_face_mode || 0;
    setCharacters((prevList) =>
      prevList.map((c) =>
        c.id === activeCharacter.id ? { ...c, face_to_face_mode: enabled ? 1 : 0 } : c,
      ),
    );
    try {
      await updateFaceToFaceMode(activeCharacter.id, enabled);
    } catch (e) {
      setCharacters((prevList) =>
        prevList.map((c) =>
          c.id === activeCharacter.id ? { ...c, face_to_face_mode: prev } : c,
        ),
      );
      setError(String(e));
    }
  }, [activeCharacter]);
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
  /** char_msg_id → log_message_id（8桁hex）のマッピング。バブルのログ折りたたみに使用する。 */
  const [msgLogIds, setMsgLogIds] = useState<Record<string, string>>({});
  /**
   * メッセージID（1on1 の char_msg_id、シナリオの turn_id）
   * → モデルへリクエストしてから応答完了までの経過時間（ミリ秒）のマッピング。
   * 現セッションのストリーミング分のみ保持し、ページリロードで消える。
   */
  const [elapsedMap, setElapsedMap] = useState<Record<string, number>>({});

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
    scenarioReasoningMap,
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
    handleScenarioYieldTo,
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
    setMsgLogIds,
  });

  /**
   * 1on1チャットの state・ストリーミング送受信ハンドラ群。
   * 共有 state（messages / sessions / selectedModel 等）の setter を渡し、1on1 専用 state は
   * フック内に閉じ込める。返り値の setReasoningMap / resetStreamingState はセッション復元/切替に使う。
   */
  const {
    streamingContent,
    streamingReasoning,
    reasoningMap,
    setReasoningMap,
    resetStreamingState: resetChatStreamingState,
    doStream,
    handleRetry,
  } = useChat({
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
    resetChatStreamingState();
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
      if (detail.model_id) {
        setSelectedModel(detail.model_id);
      }
      setMessages(detail.messages);
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
      setMsgLogIds(restoredLogIds);
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

  /** メッセージ送信。画像がある場合は先にアップロードしてから送信する。 */
  const handleSend = useCallback(async (content: string, files: File[]) => {
    if (!activeSessionId) return;

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

    setSending(true);
    try {
      await doStream(activeSessionId, content, imageIds, selectedModel || undefined);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [activeSessionId, selectedModel, doStream]);

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
            ) : activeSessionId ? (
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

            {/* 右側ボタン群: テキスト/対面切替（1on1のみ） + テーマ切り替え + エクスポート。
                シナリオの「あらすじ」ボタンはモデルチップ側のモーダルに統合済み。 */}
            <div className="pointer-events-auto ml-auto flex items-center gap-1.5">
              {activeSessionId && !isScenarioSession && activeCharacter && (
                <button
                  onClick={() => handleToggleFaceToFace(!faceToFaceMode)}
                  title={faceToFaceMode ? "テキストモードに切り替え" : "対面モードに切り替え"}
                  aria-label={faceToFaceMode ? "テキストモードに切り替え" : "対面モードに切り替え"}
                  aria-pressed={faceToFaceMode}
                  className="flex items-center justify-center rounded-lg transition-colors"
                  style={{
                    background: faceToFaceMode ? "rgba(220, 60, 80, 0.9)" : "var(--ch-bg)",
                    color: faceToFaceMode ? "#fff" : "var(--ch-t3)",
                    border: "1px solid var(--ch-sep2)",
                    boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
                    padding: "6px 8px",
                  }}
                >
                  {faceToFaceMode ? (
                    /* 対面中: user アイコン（人物） */
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={16} height={16}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
                    </svg>
                  ) : (
                    /* テキスト中: chat-bubble アイコン */
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={16} height={16}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 0 1-.825-.242m9.345-8.334a2.126 2.126 0 0 0-.476-.095 48.64 48.64 0 0 0-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.152-3.026-2.76-3.235A48.455 48.455 0 0 0 11.25 3c-2.115 0-4.198.137-6.24.402-1.608.209-2.76 1.614-2.76 3.235v6.226c0 1.621 1.152 3.026 2.76 3.235.577.075 1.157.14 1.74.194V21l4.155-4.155" />
                    </svg>
                  )}
                </button>
              )}
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
            onYieldTo={handleScenarioYieldTo}
            onEditUserTurn={handleScenarioEditUserTurn}
            onRegenerate={handleScenarioRegenerate}
            onDiscard={handleScenarioDiscard}
            onHeaderVisibilityChange={setHeaderVisible}
            elapsedMap={elapsedMap}
            scenarioReasoningMap={scenarioReasoningMap}
            synopsisBar={synopsisBar}
            synopsisGenerating={synopsisGenerating}
            onOpenSynopsisCreate={handleOpenSynopsisCreate}
            synopsisLastTurnIndex={scenarioSynopsis?.last_turn_index ?? -1}
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
            faceToFaceMode={faceToFaceMode}
            faceToFaceBgUrl={faceToFaceBgUrl}
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
              ? (activeScenarioTemplate?.pc_slots?.[0]?.name ?? userName)
              : userName
          }
          reasoningMap={reasoningMap}
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
