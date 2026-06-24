/**
 * サイドバーコンポーネント。
 * ブランド表示・新規セッション作成（モーダル起動）・セッション一覧の表示・
 * セッション削除・タイトル編集を担当する。
 */
import { useState, useRef } from "react";
import type { Model, PcAssignment, ScenarioSession, Session } from "../api";
import { charNameOf } from "../api";
import NewSessionPicker from "./NewSessionPicker";
import { CharacterAvatar } from "./ChatBubbles";

/** サイドバーで表示する統合セッション型。1on1/scenario を判別可能にする。 */
export type AnySession = Session | ScenarioSession;

interface Props {
  /** 利用可能なモデル一覧 */
  models: Model[];
  /** セッション一覧（1on1 / scenario が混在） */
  sessions: AnySession[];
  /** 現在選択中のセッションID */
  activeSessionId: string | null;
  /** サイドバーの開閉状態 */
  isOpen: boolean;
  /** サイドバー開閉トグルコールバック */
  onToggle: () => void;
  /** セッション選択時のコールバック */
  onSelectSession: (sessionId: string) => void;
  /**
   * 新規チャット作成時のコールバック。
   * @param modelId - "{char_name}@{preset_name}" 形式のモデルID。
   */
  onNewChat: (modelId: string) => void;
  /** シナリオテンプレートからプレイセッションを起動するコールバック。
   *
   * `gmPresetId` は GM プリセット（必須）、`synopsisPresetId` はあらすじ蒸留専用プリセット
   * （必須）。`engineType` は "ensemble"（既存・GMのみ）/ "ensemble_pc"（TRPG: GM+PC配役）。
   * NewSessionPicker の Scenario タブで選んだ値が渡る。
   */
  onStartScenario: (
    scenarioId: string,
    gmPresetId: string,
    synopsisPresetId: string,
    title: string | undefined,
    engineType: "ensemble" | "ensemble_pc",
    pcAssignments?: PcAssignment[],
  ) => void;
  /** セッション削除時のコールバック（session_type に応じて呼び出し側が分岐） */
  onDeleteSession: (sessionId: string) => void;
  /** セッションタイトル変更時のコールバック */
  onRenameSession: (sessionId: string, newTitle: string) => void;
}

/**
 * セッション行の先頭アイコン。
 * 1on1 はキャラクターアバター、シナリオは ✦ 円。
 *
 * アバター画像は CharacterAvatar が CharacterImageContext から自動解決する
 * （キャラクター設定に登録された画像。未登録・取得失敗時はイニシャル表示）。
 */
function SessionIcon({ session }: { session: AnySession }) {
  if (session.session_type === "scenario") {
    return (
      <span className="shrink-0 w-[18px] h-[18px] rounded-full bg-ch-s3 flex items-center justify-center text-[10px] text-ch-t2">
        ✦
      </span>
    );
  }
  // 1on1: model_id（"{char}@{preset}"）からキャラクター名を導出する。
  const charName = charNameOf((session as Session).model_id ?? "");
  return (
    <span className="shrink-0">
      <CharacterAvatar characterName={charName} size={18} />
    </span>
  );
}

/** セッション一覧と新規作成UIを提供するサイドバー。モバイルはオーバーレイ表示、デスクトップはインライン表示。 */
export default function Sidebar({
  models,
  sessions,
  activeSessionId,
  isOpen,
  onToggle: _onToggle,
  onSelectSession,
  onNewChat,
  onStartScenario,
  onDeleteSession,
  onRenameSession,
}: Props) {
  /** 削除確認中のセッションID */
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  /** インライン編集中のセッションID */
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  /** 編集中のタイトル文字列 */
  const [editingTitle, setEditingTitle] = useState("");
  /** タイトル編集インプットのref（フォーカス制御用） */
  const editInputRef = useRef<HTMLInputElement>(null);

  /** 新規セッション作成モーダルの開閉状態 */
  const [pickerOpen, setPickerOpen] = useState(false);

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
        flex flex-col bg-ch-s1 h-full w-52 shrink-0
        fixed left-0 top-0 z-30
        sm:relative sm:z-auto
        transition-transform duration-200 ease-in-out
        ${isOpen ? "translate-x-0" : "-translate-x-full sm:hidden"}
      `}
      style={{ borderRight: "1px solid var(--ch-sep)" }}
    >
      {/* ヘッダー: ブランド + 新規作成ボタン */}
      <div className="flex items-center gap-1.5 px-3.5 pt-3.5 pb-2.5">
        <span className="text-ch-t1 text-[15px] font-bold flex-1" style={{ letterSpacing: "-0.02em" }}>
          Chotgor
        </span>
        <button
          onClick={() => setPickerOpen(true)}
          title="新しい会話"
          className="text-ch-t3 hover:text-ch-t1 transition-colors leading-none text-xl px-1"
          aria-label="新しい会話を作成"
        >
          +
        </button>
      </div>

      {/* セッション一覧 */}
      <nav className="flex-1 overflow-y-auto px-1.5 pb-3">
        {sessions.length === 0 && (
          <p className="text-ch-t3 text-xs text-center mt-8">チャット履歴なし</p>
        )}
        {sessions.map((s) => {
          const active = s.id === activeSessionId;
          return (
            <div
              key={s.id}
              onClick={() => onSelectSession(s.id)}
              className={`group relative flex items-center gap-1.5 px-2.5 py-2 rounded-lg cursor-pointer transition-colors ${
                active ? "bg-ch-s2 text-ch-t1" : "text-ch-t2 hover:bg-ch-s2/60"
              }`}
            >
              {/* セッション種別アイコン */}
              <SessionIcon session={s} />

              {/* セッションタイトル（ダブルクリックでインライン編集） */}
              {editingSessionId === s.id ? (
                <input
                  ref={editInputRef}
                  className="flex-1 bg-ch-s3 text-ch-t1 text-xs rounded px-1.5 outline-none min-w-0"
                  style={{ border: "1px solid var(--ch-accent)" }}
                  value={editingTitle}
                  onChange={(e) => setEditingTitle(e.target.value)}
                  onBlur={() => handleTitleEditCommit(s.id)}
                  onKeyDown={(e) => handleTitleEditKeyDown(e, s.id)}
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <span
                  className={`flex-1 truncate text-xs ${active ? "font-medium" : ""}`}
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
                  className="opacity-0 group-hover:opacity-100 text-ch-t3 hover:text-red-500 text-xs transition-opacity shrink-0"
                >
                  ✕
                </button>
              )}
            </div>
          );
        })}
      </nav>

      {/* 新規セッション作成モーダル */}
      {pickerOpen && (
        <NewSessionPicker
          models={models}
          onClose={() => setPickerOpen(false)}
          onNewChat={onNewChat}
          onStartScenario={onStartScenario}
        />
      )}
    </aside>
  );
}
