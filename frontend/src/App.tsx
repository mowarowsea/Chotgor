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
  fetchScenarioSessions,
  fetchScenarioSession,
  fetchScenarioTurns,
  startScenarioSession,
  deleteScenarioSession,
  deleteScenarioTurnsFrom,
  streamScenarioMessage,
  fetchScenarioSynopsis,
  patchScenarioSynopsis,
  regenerateScenarioSynopsis,
  fetchScenarioPresets,
} from "./api";
import type {
  Model,
  Session,
  ChatMessage,
  StreamEvent,
  GroupStreamEvent,
  Drift,
  Character,
  ScenarioSession,
  ScenarioSynopsis,
  ScenarioTemplate,
  ScenarioPreset,
  ScenarioNpc,
  ScenarioTurn,
} from "./api";
import { charNameOf } from "./api";

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
import Sidebar from "./components/Sidebar";
import type { AnySession } from "./components/Sidebar";
import ChatView from "./components/ChatView";
import GroupChatView from "./components/GroupChatView";
import ScenarioChatView from "./components/ScenarioChatView";
import type { PendingBubble } from "./components/ScenarioChatView";
import DriftBadge from "./components/DriftBadge";
import ExportDialog from "./components/ExportDialog";
import { CharacterAvatar, CharacterImageProvider } from "./components/ChatBubbles";
import CharPresetMenu from "./components/CharPresetMenu";
import SynopsisModal from "./components/SynopsisModal";
import { useTheme } from "./hooks/useTheme";

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
  /** ライト/ダークテーマの状態と切り替え関数。 */
  const { dark, toggle: toggleTheme } = useTheme();
  /** ヘッダーのモデル切り替えメニューの開閉状態。 */
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  /** あらすじモーダルの開閉状態（シナリオセッション用）。 */
  const [synopsisModalOpen, setSynopsisModalOpen] = useState(false);
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
  /** アクティブセッションに紐づくキャラクターID（1on1チャット時のDriftBadge用）。 */
  const activeCharacterId = drifts.length > 0 ? drifts[0].character_id : "";
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
      return cfg.participants as Array<{ char_name: string; preset_name: string }>;
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
  /** グループチャット参加者のキャラクター名+ID+プリセット名リスト（ヘッダのDriftBadge用）。 */
  const groupParticipants = groupParticipantEntries.map(({ char_name, preset_name }) => ({
    charName: char_name,
    // group_config の preset_name（新セッション）→ 旧セッションはメッセージから補完。
    presetName: preset_name || groupPresetFallback.get(char_name) || "",
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
  /** 直近のグループターンで司会がエラーになったかどうか。司会再試行ボタンの表示に使用する。 */
  const [groupDirectorErrored, setGroupDirectorErrored] = useState(false);
  /** char_msg_id → log_message_id（8桁hex）のマッピング。バブルのログ折りたたみに使用する。 */
  const [msgLogIds, setMsgLogIds] = useState<Record<string, string>>({});

  /** シナリオプレイセッション一覧（サイドバーで session に混ぜる）。 */
  const [scenarioSessions, setScenarioSessions] = useState<ScenarioSession[]>([]);
  /**
   * scenarioSessions の最新値を参照するための ref。
   * 初回マウント時のハッシュ復元など、setState 直後に同期判定が必要な箇所で
   * stale な closure を避けるために使う。
   */
  const scenarioSessionsRef = useRef<ScenarioSession[]>([]);
  scenarioSessionsRef.current = scenarioSessions;
  /** 現在選択中のシナリオプレイセッション。 */
  const [activeScenarioSession, setActiveScenarioSession] = useState<ScenarioSession | null>(null);
  /** 元シナリオテンプレ（GM 設定や場所表示に使う）。 */
  const [activeScenarioTemplate, setActiveScenarioTemplate] = useState<ScenarioTemplate | null>(null);
  /** GM プリセット一覧（シナリオヘッダーの gm_preset_id → 表示名解決に使う）。 */
  const [scenarioPresets, setScenarioPresets] = useState<ScenarioPreset[]>([]);
  /** アクティブなシナリオの gm_preset_id を表示用プリセット名に解決する。 */
  const scenarioPresetName = useMemo(() => {
    const id = activeScenarioTemplate?.gm_preset_id;
    if (!id) return null;
    return scenarioPresets.find((p) => p.id === id)?.name ?? null;
  }, [activeScenarioTemplate, scenarioPresets]);
  const [scenarioNpcs, setScenarioNpcs] = useState<ScenarioNpc[]>([]);
  const [scenarioTurns, setScenarioTurns] = useState<ScenarioTurn[]>([]);
  /** ストリーミング中の未確定吹き出し列。 */
  const [scenarioPending, setScenarioPending] = useState<PendingBubble[]>([]);
  /** セッションのあらすじ（記憶捏造対策）。未取得は null。 */
  const [scenarioSynopsis, setScenarioSynopsis] = useState<ScenarioSynopsis | null>(null);

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
    setGroupDirectorErrored(false);
    // シナリオ関連の状態もリセット
    setActiveScenarioSession(null);
    setActiveScenarioTemplate(null);
    setScenarioNpcs([]);
    setScenarioTurns([]);
    setScenarioPending([]);
    setScenarioSynopsis(null);
    // URL ハッシュを更新して復帰時に同じセッションを開けるようにする
    window.location.hash = sessionId;

    // セッション種別を一覧から判別する（scenarios と sessions の両方を見る）。
    // ref 経由で参照するのは、初回マウント時のハッシュ復元で setScenarioSessions 直後に
    // この関数が呼ばれた際、state closure はまだ空 [] のままで scenario と判定できないため。
    const isScenario = scenarioSessionsRef.current.some((s) => s.id === sessionId);
    if (isScenario) {
      try {
        const [detail, ts, syn] = await Promise.all([
          fetchScenarioSession(sessionId),
          fetchScenarioTurns(sessionId),
          fetchScenarioSynopsis(sessionId).catch(() => null),
        ]);
        setActiveScenarioSession(detail);
        setActiveScenarioTemplate(detail.scenario);
        setScenarioNpcs(detail.npcs);
        setScenarioTurns(ts);
        setScenarioSynopsis(syn);
      } catch (e) {
        setError(String(e));
      }
      return;
    }

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
        await deleteScenarioSession(sessionId);
        setScenarioSessions((prev) => prev.filter((s) => s.id !== sessionId));
      } else {
        await deleteSession(sessionId);
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      }
      if (activeSessionId === sessionId) {
        window.location.hash = "";
        setActiveSessionId(null);
        setMessages([]);
        setActiveScenarioSession(null);
        setActiveScenarioTemplate(null);
        setScenarioNpcs([]);
        setScenarioTurns([]);
      }
    } catch (e) {
      setError(String(e));
    }
  }, [activeSessionId, scenarioSessions]);

  /** シナリオテンプレートからプレイセッションを起動する。 */
  const handleStartScenario = useCallback(
    async (scenarioId: string, title?: string) => {
      setError(null);
      try {
        const created = await startScenarioSession(scenarioId, title);
        setScenarioSessions((prev) => [created, ...prev]);
        // ref も即座に更新する。直後の handleSelectSession 等が ref を読むケースに備える。
        scenarioSessionsRef.current = [created, ...scenarioSessionsRef.current];
        // 起動と同時に詳細＋ターン履歴を取り直す。
        // バックエンドは scenario.intro を起動時に固定ターンとして挿入するため、
        // ここで turns を fetch しないと intro が画面に出ない（リロード後にだけ見える）。
        const [detail, initialTurns, initialSyn] = await Promise.all([
          fetchScenarioSession(created.id),
          fetchScenarioTurns(created.id),
          fetchScenarioSynopsis(created.id).catch(() => null),
        ]);
        setActiveSessionId(created.id);
        setActiveScenarioSession(detail);
        setActiveScenarioTemplate(detail.scenario);
        setScenarioNpcs(detail.npcs);
        setScenarioTurns(initialTurns);
        setScenarioPending([]);
        setScenarioSynopsis(initialSyn);
        window.location.hash = created.id;
      } catch (e) {
        setError(String(e));
      }
    },
    [],
  );

  /** シナリオセッションのフィールド更新（終了など）後に呼ばれる。 */
  const refreshScenarioSession = useCallback(async () => {
    if (!activeScenarioSession) return;
    try {
      const detail = await fetchScenarioSession(activeScenarioSession.id);
      setActiveScenarioSession(detail);
      setActiveScenarioTemplate(detail.scenario);
      setScenarioNpcs(detail.npcs);
      setScenarioSessions((prev) =>
        prev.map((s) => (s.id === detail.id ? detail : s)),
      );
    } catch (e) {
      setError(String(e));
    }
  }, [activeScenarioSession]);

  /**
   * シナリオ発話送信。SSE ストリームを消費しながら吹き出しを更新する。
   *
   * autoAdvance=true なら「ユーザは無言で続きを促す」モード。
   * content は何が来てもサーバ側で無視され、user turn も保存されない。
   */
  const handleScenarioSend = useCallback(
    async (content: string, autoAdvance: boolean = false) => {
      if (!activeScenarioSession) return;
      setError(null);
      setSending(true);
      setScenarioPending([]);
      try {
        const sessionId = activeScenarioSession.id;
        const newPending: PendingBubble[] = [];
        for await (const ev of streamScenarioMessage(sessionId, content, autoAdvance)) {
          if (ev.type === "user_saved") {
            // user_saved はユーザ発話を確定ターンとしてリストに追加する
            setScenarioTurns((prev) => [...prev, ev.turn]);
          } else if (ev.type === "speaker_start") {
            // 新しい吹き出しを未確定として追加する。
            // 安定キー (`id`) をここで一度だけ発行することで、後段の shift で
            // インデックスがズレても他の pending バブルが React 上で再マウントされない。
            newPending.push({
              id:
                typeof crypto !== "undefined" && "randomUUID" in crypto
                  ? crypto.randomUUID()
                  : `pending-${Date.now()}-${Math.random().toString(36).slice(2)}`,
              speaker_type: ev.speaker_type,
              speaker_name: ev.speaker_name,
              speaker_id: ev.speaker_id,
              is_known: ev.is_known,
              content: "",
            });
            setScenarioPending([...newPending]);
          } else if (ev.type === "content_delta") {
            // 最新の未確定吹き出しに本文を追記する
            if (newPending.length > 0) {
              newPending[newPending.length - 1].content += ev.text;
              setScenarioPending([...newPending]);
            }
          } else if (ev.type === "speaker_end") {
            // 確定ターンとしてリストに追加し、対応する未確定吹き出しを除く
            setScenarioTurns((prev) => [...prev, ev.turn]);
            if (newPending.length > 0) newPending.shift();
            setScenarioPending([...newPending]);
          } else if (ev.type === "turn_complete") {
            // ターン完了。残った未確定吹き出しは捨てる（speaker_end でほぼ消えるはず）。
            newPending.length = 0;
            setScenarioPending([]);
          } else if (ev.type === "synopsis_updated") {
            // バックエンドが履歴上限を超えた古いターンを synopsis_auto に追記した。
            // UI のあらすじパネルへ反映する。
            setScenarioSynopsis(ev.synopsis);
          } else if (ev.type === "error") {
            setError(ev.message);
            break;
          } else if (ev.type === "done") {
            break;
          }
        }
        // 完了後にサーバから真の turns を取り直して整合性確保
        if (sessionId === activeSessionIdRef.current) {
          try {
            const ts = await fetchScenarioTurns(sessionId);
            setScenarioTurns(ts);
          } catch {
            // 取得失敗は無視
          }
          // GM プリセット等の設定変更をヘッダーに反映するためテンプレを取り直す。
          fetchScenarioSession(sessionId)
            .then((d) => {
              if (sessionId === activeSessionIdRef.current) {
                setActiveScenarioTemplate(d.scenario);
              }
            })
            .catch(() => {});
          // セッション一覧も最新化（updated_at 反映）
          fetchScenarioSessions().then(setScenarioSessions).catch(() => {});
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setScenarioPending([]);
        setSending(false);
      }
    },
    [activeScenarioSession],
  );

  /**
   * シナリオの GM 応答を 1 ターン丸ごと再生成する。
   *
   * ターン境界は `raw_response` を共有する連続バブル列で判定する:
   *   - GM の 1 回の LLM 呼出 = 同一 raw_response の GM バブル列
   *   - その直前に user 発話があれば通常ターン → user 起点で再ストリーム
   *   - 直前に user 発話がなければ auto_advance ターン → GM 列の先頭から
   *     auto_advance=true で再ストリーム
   *
   * 1on1 chat の retry と同じ思想で、「ターンの開始点」より後を捨てる方式。
   */
  const handleScenarioRegenerate = useCallback(async () => {
    if (!activeScenarioSession) return;
    if (scenarioTurns.length === 0) return;

    // 末尾 GM 列の先頭 index を raw_response 連続性で探す。
    // 末尾が user の場合（GM 応答待ち状態）はそのまま user を起点にする。
    let lastTurnStart = scenarioTurns.length - 1;
    if (scenarioTurns[lastTurnStart].speaker_type !== "user") {
      const tailRaw = scenarioTurns[lastTurnStart].raw_response;
      while (lastTurnStart > 0) {
        const prev = scenarioTurns[lastTurnStart - 1];
        if (prev.speaker_type === "user") break;
        if (prev.raw_response !== tailRaw) break;
        lastTurnStart--;
      }
    }

    // 直前に user 発話があるかを見て、通常 / auto_advance を判別。
    const prev = lastTurnStart > 0 ? scenarioTurns[lastTurnStart - 1] : null;
    let pivot: ScenarioTurn;
    let resend: () => Promise<void>;
    if (prev && prev.speaker_type === "user") {
      // 通常ターン: user を含めて削除し、同じ発話で再ストリーム
      pivot = prev;
      const content = prev.content;
      resend = () => handleScenarioSend(content, false);
    } else if (scenarioTurns[lastTurnStart].speaker_type !== "user") {
      // auto_advance ターン: GM 列先頭から削除して auto_advance で再ストリーム
      pivot = scenarioTurns[lastTurnStart];
      resend = () => handleScenarioSend("", true);
    } else {
      // 末尾が user で GM 応答が無い特殊状態（前回ストリームエラー後など）
      pivot = scenarioTurns[lastTurnStart];
      const content = pivot.content;
      resend = () => handleScenarioSend(content, false);
    }

    const pivotIndex = pivot.turn_index;
    try {
      await deleteScenarioTurnsFrom(activeScenarioSession.id, pivot.id);
      setScenarioTurns((prevTurns) =>
        prevTurns.filter((t) => t.turn_index < pivotIndex),
      );
      await resend();
    } catch (e) {
      setError(String(e));
    }
  }, [activeScenarioSession, scenarioTurns, handleScenarioSend]);

  /**
   * ユーザバブルの編集確定処理。
   *
   * 編集対象 user turn 以降を全削除し、新しい内容で再ストリームする。
   * GM 応答の再生成と同じ流れ。
   */
  const handleScenarioEditUserTurn = useCallback(
    async (turnId: string, newContent: string) => {
      if (!activeScenarioSession) return;
      const target = scenarioTurns.find((t) => t.id === turnId);
      if (!target) return;
      try {
        await deleteScenarioTurnsFrom(activeScenarioSession.id, turnId);
        setScenarioTurns((prev) =>
          prev.filter((t) => t.turn_index < target.turn_index),
        );
        await handleScenarioSend(newContent);
      } catch (e) {
        setError(String(e));
      }
    },
    [activeScenarioSession, scenarioTurns, handleScenarioSend],
  );

  /**
   * グループチャットのメッセージ送信実装。
   * SSE受信中はキャラクター名を waitingCharacter で表示し、
   * 受信完了後に全メッセージをサーバーから再取得して確定する。
   *
   * @param skip - true の場合、ユーザメッセージを保存せず司会へ直接ターンを委譲する（ユーザターンスキップ）。
   * @param targetCharacter - 指定した場合、司会を介さずそのキャラクターを手動指名して発言させる。
   */
  const _doGroupStream = useCallback(async (
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
   * グループチャットの司会を手動で再試行する。
   * 司会エラー後にユーザがもう一度司会へ次発言者の判断を依頼する。
   */
  const handleGroupRetryDirector = useCallback(async () => {
    if (!activeSessionId || sending) return;
    _doGroupStream(activeSessionId, "", [], true);
  }, [activeSessionId, sending, _doGroupStream]);

  /**
   * グループチャットで任意の参加者を手動指名して発言させる。
   * 司会を介さず指定キャラクターに直接リクエストする（司会エラー時の代替手段）。
   */
  const handleGroupRequestCharacter = useCallback(async (charName: string) => {
    if (!activeSessionId || sending) return;
    _doGroupStream(activeSessionId, "", [], false, charName);
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

  /** あらすじの部分更新（auto/manual）。SynopsisModal から呼ばれる。 */
  const handleSynopsisChange = useCallback(
    async (patch: { auto?: string; manual?: string }) => {
      if (!activeScenarioSession) return;
      try {
        const updated = await patchScenarioSynopsis(activeScenarioSession.id, patch);
        setScenarioSynopsis(updated);
      } catch (e) {
        setError(String(e));
      }
    },
    [activeScenarioSession],
  );

  /** あらすじの自動追記フローを手動起動する。SynopsisModal から呼ばれる。 */
  const handleSynopsisRegenerate = useCallback(async () => {
    if (!activeScenarioSession) return;
    try {
      const updated = await regenerateScenarioSynopsis(activeScenarioSession.id);
      setScenarioSynopsis(updated);
    } catch (e) {
      setError(String(e));
    }
  }, [activeScenarioSession]);

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
        characters={characters}
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
                className="pointer-events-auto flex items-center gap-2 rounded-full bg-ch-bg min-w-0 max-w-[60%]"
                style={{ border: "1px solid var(--ch-sep2)", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", padding: "5px 12px 5px 6px" }}
              >
                <span className="shrink-0 w-6 h-6 rounded-full bg-ch-s1 flex items-center justify-center text-ch-t2 text-xs">✦</span>
                <div className="min-w-0">
                  <div className="text-ch-t1 text-[13px] font-semibold truncate leading-tight">{activeScenarioSession.title}</div>
                  {/* GM プリセット名（チャット中の変更に追従するためレスポンス毎に再取得）。 */}
                  <div className="text-ch-t3 text-[10px] font-mono leading-tight truncate">
                    {scenarioPresetName ? `@${scenarioPresetName}` : "scenario"}
                  </div>
                </div>
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
                  {groupParticipants.map(({ charName, presetName, characterId }) => {
                    const charDrifts = drifts.filter((d) => d.character_id === characterId);
                    return (
                      <div key={charName} className="flex items-center gap-1 shrink-0">
                        <span className="text-ch-t1 text-[13px] font-semibold">{charName}</span>
                        {presetName && (
                          <span className="text-ch-t3 text-[10px] font-mono">@{presetName}</span>
                        )}
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
                {activeCharacterId && (
                  <DriftBadge
                    drifts={drifts}
                    sessionId={activeSessionId}
                    characterId={activeCharacterId}
                    onDriftsChange={() => refreshDrifts(activeSessionId)}
                  />
                )}
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

            {/* 右側ボタン群: あらすじ（シナリオ時）+ テーマ切り替え + エクスポート（ピルボタン） */}
            <div className="pointer-events-auto ml-auto flex items-center gap-1.5">
              {activeSessionId && isScenarioSession && activeScenarioSession && (
                <button
                  onClick={() => setSynopsisModalOpen(true)}
                  title="これまでのあらすじ"
                  className="flex items-center gap-1 rounded-lg bg-ch-bg text-ch-t2 hover:text-ch-t1 transition-colors text-xs"
                  style={{ border: "1px solid var(--ch-sep2)", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", padding: "5px 10px" }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={14} height={14}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 0 0 6 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 0 1 6 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 0 1 6-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0 0 18 18a8.967 8.967 0 0 0-6 2.292m0-14.25v14.25" />
                  </svg>
                  あらすじ
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
            onEditUserTurn={handleScenarioEditUserTurn}
            onRegenerate={handleScenarioRegenerate}
            onHeaderVisibilityChange={setHeaderVisible}
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

      {/* あらすじモーダル（シナリオセッション用） */}
      {synopsisModalOpen && isScenarioSession && activeScenarioSession && (
        <SynopsisModal
          synopsis={scenarioSynopsis}
          onChange={handleSynopsisChange}
          onRegenerate={handleSynopsisRegenerate}
          disabled={sending}
          onClose={() => setSynopsisModalOpen(false)}
        />
      )}
    </div>
    </CharacterImageProvider>
  );
}
