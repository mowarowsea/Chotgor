/**
 * サイドバーコンポーネント。
 * セッション一覧の表示・新規チャット開始・グループチャット作成・セッション削除を担当する。
 */
import { useState, useEffect, useRef } from "react";
import type { Model, Session } from "../api";

interface Props {
  /** 利用可能なモデル一覧 */
  models: Model[];
  /** セッション一覧 */
  sessions: Session[];
  /** 現在選択中のセッションID */
  activeSessionId: string | null;
  /** サイドバーの開閉状態 */
  isOpen: boolean;
  /** サイドバー開閉トグルコールバック */
  onToggle: () => void;
  /** セッション選択時のコールバック */
  onSelectSession: (sessionId: string) => void;
  /** 新規チャット作成時のコールバック */
  onNewChat: (modelId: string) => void;
  /** 新規グループチャット作成時のコールバック */
  onNewGroupChat: (participants: string[], directorModelId: string, maxAutoTurns: number) => void;
  /** セッション削除時のコールバック */
  onDeleteSession: (sessionId: string) => void;
}

/** セッション一覧とモデル選択UIを提供するサイドバー。モバイルはオーバーレイ表示、デスクトップはインライン表示。 */
export default function Sidebar({
  models,
  sessions,
  activeSessionId,
  isOpen,
  onToggle,
  onSelectSession,
  onNewChat,
  onNewGroupChat,
  onDeleteSession,
}: Props) {
  /** 新規チャットに使うモデルID */
  const [selectedModel, setSelectedModel] = useState(models[0]?.id ?? "");

  /** モデル一覧が初めてロードされたとき、選択モデルを先頭に設定する */
  const hasSetInitialModel = useRef(false);
  useEffect(() => {
    if (models.length > 0 && !hasSetInitialModel.current) {
      hasSetInitialModel.current = true;
      setSelectedModel(models[0].id);
    }
  }, [models]);

  /** 削除確認中のセッションID */
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  /** グループチャット作成パネルの開閉状態 */
  const [groupPanelOpen, setGroupPanelOpen] = useState(false);
  /** グループチャット参加者として選択されたモデルIDセット */
  const [groupSelected, setGroupSelected] = useState<Set<string>>(new Set());
  /** 司会役モデルID ("{char_name}@{preset_name}" 形式) */
  const [directorModelId, setDirectorModelId] = useState("");
  /** 最大自動ターン数 */
  const [maxAutoTurns, setMaxAutoTurns] = useState(3);

  /** モデルIDからキャラクター名を抽出する */
  const charNameOf = (modelId: string) => modelId.split("@")[0];

  const handleNewChat = () => {
    if (!selectedModel) return;
    onNewChat(selectedModel);
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

  return (
    /*
     * モバイル: fixed でオーバーレイ表示。isOpen に応じて translateX でスライドイン/アウト。
     * デスクトップ (sm+): relative でインライン表示。isOpen が false のときは hidden で非表示。
     */
    <aside
      className={`
        flex flex-col bg-zinc-900 border-r border-zinc-800 h-full w-64 shrink-0
        fixed left-0 top-0 z-30
        sm:relative sm:z-auto
        transition-transform duration-200 ease-in-out
        ${isOpen ? "translate-x-0" : "-translate-x-full sm:hidden"}
      `}
    >
      {/* ヘッダー */}
      <div className="p-4 border-b border-zinc-800">
        <h1 className="text-lg font-bold text-zinc-100 mb-3">Chotgor</h1>

        {/* モデル選択 */}
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full bg-zinc-800 text-zinc-100 text-sm rounded px-2 py-1.5 mb-2 border border-zinc-700 focus:outline-none focus:border-zinc-500"
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
          onClick={handleNewChat}
          disabled={!selectedModel}
          className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-medium rounded px-3 py-1.5 transition-colors"
        >
          + 新規チャット
        </button>

        {/* グループチャット作成ボタン */}
        <button
          onClick={() => setGroupPanelOpen((o) => !o)}
          className="w-full bg-zinc-700 hover:bg-zinc-600 text-zinc-200 text-sm font-medium rounded px-3 py-1.5 transition-colors mt-1 flex items-center justify-center gap-1"
        >
          <span>👥</span>
          <span>グループチャット</span>
          <span className="ml-auto text-xs">{groupPanelOpen ? "▲" : "▼"}</span>
        </button>

        {/* グループチャット作成パネル */}
        {groupPanelOpen && (
          <div className="mt-2 space-y-2 border border-zinc-700 rounded-lg p-3 bg-zinc-800/50">
            {/* 参加者選択 */}
            <p className="text-zinc-400 text-xs font-medium">参加者を選択（2名以上）</p>
            <div className="space-y-1 max-h-36 overflow-y-auto">
              {models.length === 0 && (
                <p className="text-zinc-500 text-xs">モデルがありません</p>
              )}
              {models.map((m) => (
                <label key={m.id} className="flex items-center gap-2 cursor-pointer group/check">
                  <input
                    type="checkbox"
                    checked={groupSelected.has(m.id)}
                    onChange={() => toggleGroupModel(m.id)}
                    className="accent-indigo-500"
                  />
                  <span className="text-zinc-300 text-xs truncate group/check-hover:text-zinc-100">
                    {charNameOf(m.id)}
                    <span className="text-zinc-500 ml-1">@{m.id.split("@")[1]}</span>
                  </span>
                </label>
              ))}
            </div>

            {/* 司会役キャラクター選択 */}
            <div>
              <p className="text-zinc-400 text-xs font-medium mb-1">司会役</p>
              <select
                value={directorModelId}
                onChange={(e) => setDirectorModelId(e.target.value)}
                className="w-full bg-zinc-700 text-zinc-100 text-xs rounded px-2 py-1.5 border border-zinc-600 focus:outline-none focus:border-zinc-400"
              >
                <option value="">司会役を選択...</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {charNameOf(m.id)}
                    <span> @{m.id.split("@")[1]}</span>
                  </option>
                ))}
              </select>
            </div>

            {/* 最大自動ターン数 */}
            <div>
              <p className="text-zinc-400 text-xs font-medium mb-1">
                最大自動ターン数: <span className="text-zinc-200">{maxAutoTurns}</span>
                {maxAutoTurns >= 5 && <span className="text-amber-400 ml-1">⚠ API消費増</span>}
              </p>
              <input
                type="range"
                min={1}
                max={10}
                value={maxAutoTurns}
                onChange={(e) => setMaxAutoTurns(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
            </div>

            {/* 作成ボタン */}
            <button
              onClick={handleCreateGroup}
              disabled={groupSelected.size < 2 || !directorModelId}
              className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs font-medium rounded px-3 py-1.5 transition-colors"
            >
              {groupSelected.size < 2
                ? "2名以上選択してください"
                : !directorModelId
                ? "司会役を選択してください"
                : `グループチャット開始 (${groupSelected.size}名)`}
            </button>
          </div>
        )}
      </div>

      {/* セッション一覧 */}
      <nav className="flex-1 overflow-y-auto py-2">
        {sessions.length === 0 && (
          <p className="text-zinc-500 text-sm text-center mt-8">チャット履歴なし</p>
        )}
        {sessions.map((s) => (
          <div
            key={s.id}
            onClick={() => onSelectSession(s.id)}
            className={`group relative flex items-center gap-2 px-4 py-2.5 cursor-pointer transition-colors ${
              s.id === activeSessionId
                ? "bg-zinc-700 text-zinc-100"
                : "hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100"
            }`}
          >
            {/* グループチャットにはアイコンを付与する */}
            {s.session_type === "group" && (
              <span className="text-xs shrink-0">👥</span>
            )}
            {/* セッションタイトル */}
            <span className="flex-1 truncate text-sm">{s.title}</span>

            {/* 削除ボタン（hover時表示） */}
            {confirmDeleteId === s.id ? (
              <span className="flex gap-1 text-xs">
                <button
                  onClick={(e) => handleConfirmDelete(e, s.id)}
                  className="text-red-400 hover:text-red-300"
                >
                  削除
                </button>
                <button
                  onClick={handleCancelDelete}
                  className="text-zinc-400 hover:text-zinc-300"
                >
                  ✕
                </button>
              </span>
            ) : (
              <button
                onClick={(e) => handleDeleteClick(e, s.id)}
                className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-zinc-500 hover:text-red-400 text-xs transition-opacity"
              >
                🗑
              </button>
            )}
          </div>
        ))}
      </nav>
    </aside>
  );
}
