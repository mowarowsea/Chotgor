/**
 * サイドバーコンポーネント。
 * セッション一覧の表示・新規チャット開始・グループチャット作成・セッション削除・タイトル編集を担当する。
 */
import { useState, useRef, useEffect } from "react";
import type { Character, Model, ScenarioSession, ScenarioTemplate, Session } from "../api";
import { charNameOf, presetNameOf, fetchScenarioTemplates } from "../api";

/** サイドバーで表示する統合セッション型。1on1/group/scenario を判別可能にする。 */
export type AnySession = Session | ScenarioSession;

interface Props {
  /** 利用可能なモデル一覧 */
  models: Model[];
  /** キャラクター一覧（Afterglowデフォルト値の取得に使用） */
  characters: Character[];
  /** セッション一覧（1on1 / group / scenario が混在） */
  sessions: AnySession[];
  /** 現在選択中のセッションID */
  activeSessionId: string | null;
  /** 現在選択中のモデルID（Appから制御） */
  selectedModel: string;
  /** モデル選択変更コールバック */
  onModelChange: (modelId: string) => void;
  /** サイドバーの開閉状態 */
  isOpen: boolean;
  /** サイドバー開閉トグルコールバック */
  onToggle: () => void;
  /** セッション選択時のコールバック */
  onSelectSession: (sessionId: string) => void;
  /**
   * 新規チャット作成時のコールバック。
   * @param modelId - "{char_name}@{preset_name}" 形式のモデルID。
   * @param afterglow - Afterglow（感情継続機構）を有効にする場合は true。
   */
  onNewChat: (modelId: string, afterglow: boolean) => void;
  /** 新規グループチャット作成時のコールバック */
  onNewGroupChat: (participants: string[], directorModelId: string, maxAutoTurns: number) => void;
  /** シナリオテンプレートからプレイセッションを起動するコールバック */
  onStartScenario: (scenarioId: string, title?: string) => void;
  /** セッション削除時のコールバック（session_type に応じて呼び出し側が分岐） */
  onDeleteSession: (sessionId: string) => void;
  /** セッションタイトル変更時のコールバック */
  onRenameSession: (sessionId: string, newTitle: string) => void;
}

/** セッション一覧とモデル選択UIを提供するサイドバー。モバイルはオーバーレイ表示、デスクトップはインライン表示。 */
export default function Sidebar({
  models,
  characters,
  sessions,
  activeSessionId,
  selectedModel,
  onModelChange,
  isOpen,
  onToggle: _onToggle,
  onSelectSession,
  onNewChat,
  onNewGroupChat,
  onStartScenario,
  onDeleteSession,
  onRenameSession,
}: Props) {
  /** アクティブセッションが1on1でない（=group or scenario）かどうか。モデルセレクタの無効化判定用。 */
  const activeType = sessions.find((s) => s.id === activeSessionId)?.session_type;
  const isGroupSession = activeType === "group" || activeType === "scenario";

  /** 削除確認中のセッションID */
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  /** インライン編集中のセッションID */
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  /** 編集中のタイトル文字列 */
  const [editingTitle, setEditingTitle] = useState("");
  /** タイトル編集インプットのref（フォーカス制御用） */
  const editInputRef = useRef<HTMLInputElement>(null);

  /** タイトルのダブルクリック時にインライン編集モードを開始する。 */
  const handleTitleDoubleClick = (e: React.MouseEvent, sessionId: string, currentTitle: string) => {
    e.stopPropagation();
    setEditingSessionId(sessionId);
    setEditingTitle(currentTitle);
    // DOMが更新された後にフォーカスを当てる
    setTimeout(() => editInputRef.current?.select(), 0);
  };

  /** タイトル編集を確定してコールバックを呼ぶ。 */
  const handleTitleEditCommit = (sessionId: string) => {
    const trimmed = editingTitle.trim();
    if (trimmed) {
      onRenameSession(sessionId, trimmed);
    }
    setEditingSessionId(null);
  };

  /** Enterで確定・Escapeでキャンセル。 */
  const handleTitleEditKeyDown = (e: React.KeyboardEvent, sessionId: string) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleTitleEditCommit(sessionId);
    } else if (e.key === "Escape") {
      setEditingSessionId(null);
    }
  };

  /** 新規チャット作成パネルの開閉状態 */
  const [newChatPanelOpen, setNewChatPanelOpen] = useState(false);
  /**
   * Afterglow（感情継続機構）チェックボックスの現在値。
   * selectedModel が変わるたびにキャラのデフォルト値で初期化する。
   */
  const [afterglowEnabled, setAfterglowEnabled] = useState(false);

  /** selectedModel が変わったとき、Afterglow チェックボックスのデフォルト値を更新する。 */
  useEffect(() => {
    const charName = charNameOf(selectedModel);
    const char = characters.find((c) => c.name === charName);
    setAfterglowEnabled(char?.afterglow_default ?? false);
  }, [selectedModel, characters]);

  /** グループチャット作成パネルの開閉状態 */
  const [groupPanelOpen, setGroupPanelOpen] = useState(false);
  /** グループチャット参加者として選択されたモデルIDセット */
  const [groupSelected, setGroupSelected] = useState<Set<string>>(new Set());
  /** 司会役モデルID ("{char_name}@{preset_name}" 形式) */
  const [directorModelId, setDirectorModelId] = useState("");
  /** 最大自動ターン数 */
  const [maxAutoTurns, setMaxAutoTurns] = useState(3);

  /** シナリオ開始パネルの開閉状態 */
  const [scenarioPanelOpen, setScenarioPanelOpen] = useState(false);
  /** 起動対象シナリオテンプレート ID */
  const [scSelectedId, setScSelectedId] = useState("");
  /** 起動時のセッションタイトル（空欄ならテンプレ title をコピー） */
  const [scTitle, setScTitle] = useState("");
  /** シナリオテンプレート一覧（パネル開時に取得） */
  const [templates, setTemplates] = useState<ScenarioTemplate[]>([]);
  useEffect(() => {
    if (scenarioPanelOpen) {
      // パネルを開くたびに最新一覧を取り直す（backend UI で追加・編集されている可能性）
      fetchScenarioTemplates().then(setTemplates).catch(() => setTemplates([]));
    }
  }, [scenarioPanelOpen]);

  /** 新規チャットを作成する。 */
  const handleNewChat = () => {
    if (!selectedModel) return;
    onNewChat(selectedModel, afterglowEnabled);
    setNewChatPanelOpen(false);
  };

  /** グループチャット参加者の選択をトグルする */
  const toggleGroupModel = (modelId: string) => {
    setGroupSelected((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) {
        next.delete(modelId);
      } else {
        next.add(modelId);
      }
      return next;
    });
  };

  /** グループチャットを作成する */
  const handleCreateGroup = () => {
    if (groupSelected.size < 2 || !directorModelId) return;
    onNewGroupChat([...groupSelected], directorModelId, maxAutoTurns);
    setGroupPanelOpen(false);
    setGroupSelected(new Set());
    setDirectorModelId("");
  };

  /** 選択中のテンプレートからプレイセッションを起動する。 */
  const handleStartScenario = () => {
    if (!scSelectedId) return;
    const title = scTitle.trim() || undefined;
    onStartScenario(scSelectedId, title);
    setScenarioPanelOpen(false);
    setScSelectedId("");
    setScTitle("");
  };

  const handleDeleteClick = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    setConfirmDeleteId(sessionId);
  };

  const handleConfirmDelete = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    setConfirmDeleteId(null);
    onDeleteSession(sessionId);
  };

  const handleCancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setConfirmDeleteId(null);
  };

  /* ── accent button style ── */
  const btnAccent = "w-full bg-ch-accent-dim border border-ch-accent/30 text-ch-accent-t text-xs font-medium rounded px-3 py-1.5 transition-colors hover:bg-ch-accent/20 hover:border-ch-accent/50 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-between";
  const btnGhost  = "w-full bg-transparent border border-ch-s3 text-ch-t3 text-xs font-medium rounded px-3 py-1.5 transition-colors hover:border-ch-s3 hover:text-ch-t2 flex items-center justify-between";

  return (
    /*
     * モバイル: fixed でオーバーレイ表示。isOpen に応じて translateX でスライドイン/アウト。
     * デスクトップ (sm+): relative でインライン表示。isOpen が false のときは hidden で非表示。
     */
    <aside
      className={`
        flex flex-col bg-ch-s1 h-full w-56 shrink-0
        fixed left-0 top-0 z-30
        sm:relative sm:z-auto
        transition-transform duration-200 ease-in-out
        ${isOpen ? "translate-x-0" : "-translate-x-full sm:hidden"}
      `}
      style={{ borderRight: "1px solid rgba(255,255,255,0.09)" }}
    >
      {/* ヘッダー */}
      <div className="px-4 pt-4 pb-3" style={{ borderBottom: "1px solid rgba(255,255,255,0.09)" }}>
        {/* ブランド */}
        <div className="mb-3">
          <span
            className="text-ch-accent-t text-xs font-semibold tracking-widest uppercase"
            style={{ textShadow: "0 0 20px rgba(106,168,130,0.35)", letterSpacing: "0.22em" }}
          >
            Chotgor
          </span>
        </div>

        {/* モデル選択 */}
        <select
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          disabled={isGroupSession}
          className="w-full bg-ch-bg text-ch-t1 text-xs rounded px-2 py-1.5 mb-2 focus:outline-none disabled:opacity-30 disabled:cursor-not-allowed appearance-none"
          style={{
            border: "1px solid rgba(255,255,255,0.16)",
            backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a5e55'/%3E%3C/svg%3E\")",
            backgroundRepeat: "no-repeat",
            backgroundPosition: "right 0.5rem center",
            paddingRight: "1.75rem",
          }}
        >
          {models.length === 0 && (
            <option value="">モデルなし</option>
          )}
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id}
            </option>
          ))}
        </select>

        {/* 新規チャットボタン */}
        <button
          onClick={() => setNewChatPanelOpen((o) => !o)}
          disabled={!selectedModel}
          className={btnAccent}
        >
          <span>+ 新規チャット</span>
          <span className="text-[10px] opacity-60">{newChatPanelOpen ? "▲" : "▼"}</span>
        </button>

        {/* 新規チャット作成パネル（Afterglow設定） */}
        {newChatPanelOpen && (
          <div className="mt-1 space-y-2 rounded p-2.5" style={{ border: "1px solid rgba(255,255,255,0.12)", background: "rgba(10,10,10,0.7)" }}>
            <label className="flex items-start gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={afterglowEnabled}
                onChange={(e) => setAfterglowEnabled(e.target.checked)}
                className="mt-0.5 shrink-0 accent-ch-accent"
              />
              <span className="text-ch-t2 text-xs leading-snug">
                前回の流れを引き継ぐ
                <span className="block text-ch-t3 text-xs mt-0.5">
                  直近5ターンの会話を前置きとして引き継ぎます
                </span>
              </span>
            </label>
            <button
              onClick={handleNewChat}
              disabled={!selectedModel}
              className={btnAccent}
            >
              <span>チャット開始</span>
            </button>
          </div>
        )}

        {/* グループチャット作成ボタン */}
        <button
          onClick={() => setGroupPanelOpen((o) => !o)}
          className={`${btnGhost} mt-1.5`}
        >
          <span className="flex items-center gap-1.5">
            <span className="text-ch-t3 text-xs">👥</span>
            <span>グループチャット</span>
          </span>
          <span className="text-[10px] opacity-60">{groupPanelOpen ? "▲" : "▼"}</span>
        </button>

        {/* グループチャット作成パネル */}
        {groupPanelOpen && (
          <div className="mt-1 space-y-2.5 rounded p-2.5" style={{ border: "1px solid rgba(255,255,255,0.12)", background: "rgba(10,10,10,0.7)" }}>
            <p className="text-ch-t3 text-xs">参加者を選択（2名以上）</p>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {models.length === 0 && (
                <p className="text-ch-t4 text-xs">モデルがありません</p>
              )}
              {models.map((m) => (
                <label key={m.id} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={groupSelected.has(m.id)}
                    onChange={() => toggleGroupModel(m.id)}
                    className="accent-ch-accent"
                  />
                  <span className="text-ch-t2 text-xs truncate">
                    {charNameOf(m.id)}
                    <span className="text-ch-t3 ml-1">@{presetNameOf(m.id)}</span>
                  </span>
                </label>
              ))}
            </div>

            <div>
              <p className="text-ch-t3 text-xs mb-1">司会役</p>
              <select
                value={directorModelId}
                onChange={(e) => setDirectorModelId(e.target.value)}
                className="w-full bg-ch-bg text-ch-t1 text-xs rounded px-2 py-1.5 focus:outline-none appearance-none"
                style={{ border: "1px solid rgba(255,255,255,0.16)" }}
              >
                <option value="">司会役を選択...</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {charNameOf(m.id)} @{presetNameOf(m.id)}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <p className="text-ch-t3 text-xs mb-1">
                最大自動ターン数: <span className="text-ch-t1">{maxAutoTurns}</span>
                {maxAutoTurns >= 5 && <span className="text-amber-600 ml-1 text-[10px]">⚠ API消費増</span>}
              </p>
              <input
                type="range"
                min={1}
                max={10}
                value={maxAutoTurns}
                onChange={(e) => setMaxAutoTurns(Number(e.target.value))}
                className="w-full accent-ch-accent"
              />
            </div>

            <button
              onClick={handleCreateGroup}
              disabled={groupSelected.size < 2 || !directorModelId}
              className={btnAccent}
            >
              <span>
                {groupSelected.size < 2
                  ? "2名以上選択してください"
                  : !directorModelId
                  ? "司会役を選択してください"
                  : `グループ開始 (${groupSelected.size}名)`}
              </span>
            </button>
          </div>
        )}

        {/* シナリオ開始ボタン */}
        <button
          onClick={() => setScenarioPanelOpen((o) => !o)}
          className={`${btnGhost} mt-1.5`}
        >
          <span className="flex items-center gap-1.5">
            <span className="text-ch-t3 text-xs">📖</span>
            <span>シナリオ</span>
          </span>
          <span className="text-[10px] opacity-60">{scenarioPanelOpen ? "▲" : "▼"}</span>
        </button>

        {/* シナリオ開始パネル — テンプレ選択 + タイトル + 開始ボタン */}
        {scenarioPanelOpen && (
          <div
            className="mt-1 space-y-2 rounded p-2.5"
            style={{ border: "1px solid rgba(255,255,255,0.12)", background: "rgba(10,10,10,0.7)" }}
          >
            <div>
              <p className="text-ch-t3 text-xs mb-1">シナリオを選択</p>
              <select
                value={scSelectedId}
                onChange={(e) => setScSelectedId(e.target.value)}
                className="w-full bg-ch-bg text-ch-t1 text-xs rounded px-2 py-1.5 focus:outline-none appearance-none"
                style={{ border: "1px solid rgba(255,255,255,0.16)" }}
              >
                <option value="">— select scenario —</option>
                {templates.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.title}
                  </option>
                ))}
              </select>
              {templates.length === 0 && (
                <p className="text-ch-t4 text-[10px] mt-1">
                  シナリオがありません。
                  <a
                    href="/ui/scenarios/new"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline ml-1 hover:text-ch-t2"
                  >
                    backend で登録
                  </a>
                </p>
              )}
            </div>
            <div>
              <p className="text-ch-t3 text-xs mb-1">セッションタイトル（任意）</p>
              <input
                value={scTitle}
                onChange={(e) => setScTitle(e.target.value)}
                placeholder="空欄ならシナリオ名をコピー"
                className="w-full bg-ch-bg text-ch-t1 text-xs rounded px-2 py-1.5 focus:outline-none"
                style={{ border: "1px solid rgba(255,255,255,0.16)" }}
              />
            </div>
            <button
              onClick={handleStartScenario}
              disabled={!scSelectedId}
              className={btnAccent}
            >
              <span>
                {!scSelectedId ? "シナリオを選択してください" : "シナリオを開始"}
              </span>
            </button>
            <p className="text-ch-t4 text-[10px]">
              シナリオの編集・NPC追加は
              <a
                href="/ui/scenarios"
                target="_blank"
                rel="noopener noreferrer"
                className="underline mx-1 hover:text-ch-t2"
              >
                backend Scenarios
              </a>
              で行います
            </p>
          </div>
        )}
      </div>

      {/* セッション一覧 */}
      <nav className="flex-1 overflow-y-auto py-1">
        {sessions.length === 0 && (
          <p className="text-ch-t4 text-xs text-center mt-8">チャット履歴なし</p>
        )}
        {sessions.map((s) => (
          <div
            key={s.id}
            onClick={() => onSelectSession(s.id)}
            className={`group relative flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors ${
              s.id === activeSessionId
                ? "bg-ch-s2 text-ch-t1"
                : "text-ch-t3 hover:bg-ch-s1 hover:text-ch-t2"
            }`}
          >
            {/* セッション種別アイコン */}
            {s.session_type === "group" && (
              <span className="text-[11px] shrink-0 opacity-50">👥</span>
            )}
            {s.session_type === "scenario" && (
              <span className="text-[11px] shrink-0 opacity-50">📖</span>
            )}
            {/* アクティブインジケーター */}
            {s.id === activeSessionId && (
              <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-full bg-ch-accent-t" />
            )}

            {/* セッションタイトル（ダブルクリックでインライン編集） */}
            {editingSessionId === s.id ? (
              <input
                ref={editInputRef}
                className="flex-1 bg-ch-s3 text-ch-t1 text-xs rounded px-1.5 outline-none min-w-0"
                style={{ border: "1px solid rgba(255,255,255,0.25)" }}
                value={editingTitle}
                onChange={(e) => setEditingTitle(e.target.value)}
                onBlur={() => handleTitleEditCommit(s.id)}
                onKeyDown={(e) => handleTitleEditKeyDown(e, s.id)}
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <span
                className="flex-1 truncate text-xs"
                onDoubleClick={(e) => handleTitleDoubleClick(e, s.id, s.title)}
              >
                {s.title}
              </span>
            )}

            {/* 削除ボタン（hover時表示） */}
            {confirmDeleteId === s.id ? (
              <span className="flex gap-1 text-[11px]">
                <button
                  onClick={(e) => handleConfirmDelete(e, s.id)}
                  className="text-red-500 hover:text-red-400"
                >
                  削除
                </button>
                <button
                  onClick={handleCancelDelete}
                  className="text-ch-t3 hover:text-ch-t2"
                >
                  ✕
                </button>
              </span>
            ) : (
              <button
                onClick={(e) => handleDeleteClick(e, s.id)}
                className="opacity-0 group-hover:opacity-100 text-ch-t4 hover:text-red-500 text-xs transition-opacity"
              >
                ✕
              </button>
            )}
          </div>
        ))}
      </nav>
    </aside>
  );
}
