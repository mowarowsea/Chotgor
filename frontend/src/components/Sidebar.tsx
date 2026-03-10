/**
 * サイドバーコンポーネント。
 * セッション一覧の表示・新規チャット開始・セッション削除を担当する。
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

  const handleNewChat = () => {
    if (!selectedModel) return;
    onNewChat(selectedModel);
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
