/**
 * DriftBadge — SELF_DRIFT表示・管理コンポーネント。
 *
 * キャラクター名の横に配置するアイコンバッジ。
 * クリック/タップでフロートパネルを開き、drift一覧のON/OFFトグルとリセットを提供する。
 *
 * フロートパネルは ReactDOM.createPortal で document.body に直接マウントすることで、
 * 祖先の overflow:hidden / backdrop-filter によるスタッキングコンテキスト問題を回避する。
 */
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Drift } from "../api";
import { toggleDrift, resetDrifts } from "../api";

interface Props {
  /** 表示対象のdrift一覧 */
  drifts: Drift[];
  /** セッションID */
  sessionId: string;
  /** キャラクターID（リセット時に使用） */
  characterId: string;
  /** drift一覧が変化した後に呼ばれるコールバック */
  onDriftsChange: () => void;
}

/** SELF_DRIFTバッジとフロートパネルコンポーネント。 */
export default function DriftBadge({ drifts, sessionId, characterId, onDriftsChange }: Props) {
  const [open, setOpen] = useState(false);
  /** バッジボタンの参照（パネル表示位置の計算に使用） */
  const buttonRef = useRef<HTMLButtonElement>(null);
  /** フロートパネルの参照（パネル外クリック検知に使用） */
  const panelRef = useRef<HTMLDivElement>(null);
  /** フロートパネルの表示位置（fixed座標） */
  const [panelPos, setPanelPos] = useState({ top: 0, left: 0 });

  /** パネル外クリックで閉じる。バッジボタンとパネルの両方を除外する。 */
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      const clickedButton = buttonRef.current?.contains(target);
      const clickedPanel = panelRef.current?.contains(target);
      if (!clickedButton && !clickedPanel) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  /** enabledなdriftの件数 */
  const activeCount = drifts.filter((d) => d.enabled).length;

  /**
   * バッジクリック時にパネルを開閉する。
   * 開く際はバッジの位置を取得してfixedパネルの座標を計算する。
   */
  const handleToggleOpen = () => {
    if (!open && buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      const panelWidth = 288;
      const margin = 8;
      const clampedLeft = Math.min(rect.left, window.innerWidth - panelWidth - margin);
      setPanelPos({ top: rect.bottom + 4, left: Math.max(margin, clampedLeft) });
    }
    setOpen((o) => !o);
  };

  const handleToggle = async (drift: Drift) => {
    try {
      await toggleDrift(sessionId, drift.id);
      onDriftsChange();
    } catch {
      // エラーは無視
    }
  };

  const handleReset = async () => {
    try {
      await resetDrifts(sessionId, characterId);
      onDriftsChange();
      setOpen(false);
    } catch {
      // エラーは無視
    }
  };

  if (drifts.length === 0) return null;

  return (
    <>
      {/* バッジアイコン */}
      <button
        ref={buttonRef}
        onClick={handleToggleOpen}
        title="SELF_DRIFT"
        className="flex items-center gap-1 px-1.5 py-0.5 rounded-full text-xs transition-colors text-ch-accent-t"
        style={{
          background: "rgba(77,140,103,0.12)",
          border: "1px solid rgba(72,130,96,0.22)",
        }}
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" width={13} height={13}>
          <path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 0 1-9.201 2.466l-.312-.311h2.433a.75.75 0 0 0 0-1.5H3.989a.75.75 0 0 0-.75.75v4.242a.75.75 0 0 0 1.5 0v-2.43l.31.31a7 7 0 0 0 11.712-3.138.75.75 0 0 0-1.449-.39Zm1.23-3.723a.75.75 0 0 0 .219-.53V2.929a.75.75 0 0 0-1.5 0V5.36l-.31-.31A7 7 0 0 0 3.239 8.188a.75.75 0 1 0 1.448.389A5.5 5.5 0 0 1 13.89 6.11l.311.31h-2.432a.75.75 0 0 0 0 1.5h4.243a.75.75 0 0 0 .53-.219Z" clipRule="evenodd" />
        </svg>
        <span>{activeCount}/{drifts.length}</span>
      </button>

      {/*
       * フロートパネル: createPortal で document.body に直接マウントする。
       * これにより、祖先要素の stacking context の影響を受けずに最前面に表示できる。
       */}
      {open && createPortal(
        <div
          ref={panelRef}
          className="fixed z-[9999] w-72 bg-ch-s2 rounded-xl shadow-2xl p-3 flex flex-col gap-2"
          style={{
            top: panelPos.top,
            left: panelPos.left,
            border: "1px solid rgba(255,255,255,0.10)",
          }}
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-ch-accent-t uppercase tracking-widest opacity-70">SELF DRIFT</span>
            {drifts.length > 0 && (
              <button
                onClick={handleReset}
                className="text-xs text-red-400/70 hover:text-red-300 transition-colors"
              >
                全リセット
              </button>
            )}
          </div>

          <ul className="flex flex-col gap-1.5">
            {drifts.map((drift) => (
              <li key={drift.id} className="flex items-start gap-2">
                {/* ON/OFFトグル */}
                <button
                  onClick={() => handleToggle(drift)}
                  className="mt-0.5 shrink-0 w-8 h-4 rounded-full transition-colors"
                  style={{
                    background: drift.enabled ? "rgba(77,140,103,0.55)" : "rgba(255,255,255,0.08)",
                    border: "1px solid rgba(255,255,255,0.10)",
                  }}
                  title={drift.enabled ? "OFFにする" : "ONにする"}
                >
                  <span
                    className={`block w-3 h-3 rounded-full bg-white mx-auto transition-transform ${
                      drift.enabled ? "translate-x-2" : "-translate-x-2"
                    }`}
                    style={{ opacity: drift.enabled ? 0.9 : 0.4 }}
                  />
                </button>
                <span
                  className={`text-xs leading-relaxed break-words ${
                    drift.enabled ? "text-ch-t1" : "text-ch-t3 line-through"
                  }`}
                >
                  {drift.content}
                </span>
              </li>
            ))}
          </ul>
        </div>,
        document.body
      )}
    </>
  );
}
