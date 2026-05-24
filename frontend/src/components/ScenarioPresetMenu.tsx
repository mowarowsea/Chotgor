/**
 * シナリオセッション用 GM プリセット切替メニュー。
 *
 * 1on1 の CharPresetMenu と同じ操作感だが、シナリオには Character の概念が無いので
 * Preset のみのフラットな選択リストにする。
 *
 * チャットヘッダー左上のチップをクリックすると開き、選択 → 適用で
 * `ScenarioSession.gm_preset_id` を PATCH 更新する（次ターン以降に反映）。
 */
import { useEffect, useRef, useState } from "react";
import type { ScenarioPreset } from "../api";

interface Props {
  /** 選択候補となる GM プリセット一覧（fetchScenarioPresets の結果）。 */
  presets: ScenarioPreset[];
  /** 現在セッションに紐付いているプリセット ID。 */
  currentPresetId: string;
  /** プリセット確定時のコールバック。サーバへ反映するのは呼び出し元の責務。 */
  onApply: (presetId: string) => void;
  /** メニューを閉じるコールバック。 */
  onClose: () => void;
}

/** シナリオセッション用の GM プリセット切替メニュー本体。 */
export default function ScenarioPresetMenu({
  presets,
  currentPresetId,
  onApply,
  onClose,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [selPresetId, setSelPresetId] = useState(currentPresetId);

  // メニュー外クリック・Esc で閉じる（CharPresetMenu と同じ挙動）。
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  /** 選択中のプリセットを確定する。 */
  const apply = () => {
    if (!selPresetId) return;
    onApply(selPresetId);
    onClose();
  };

  return (
    <div
      ref={ref}
      className="absolute left-0 top-[calc(100%+6px)] z-50 bg-ch-bg rounded-xl p-3"
      style={{
        border: "1px solid var(--ch-sep2)",
        boxShadow: "var(--ch-shadow)",
        width: 240,
      }}
    >
      <div
        className="text-[10px] text-ch-t3 font-mono mb-1.5"
        style={{ letterSpacing: "0.06em" }}
      >
        GM MODEL <span className="opacity-60">· {presets.length}</span>
      </div>
      {presets.length === 0 ? (
        <p className="text-ch-t3 text-xs mb-3">
          LLM プリセットがありません
        </p>
      ) : (
        <div className="flex flex-col gap-0.5 mb-3 max-h-56 overflow-y-auto">
          {presets.map((p) => {
            const active = selPresetId === p.id;
            return (
              <button
                key={p.id}
                onClick={() => setSelPresetId(p.id)}
                className="text-left rounded-md px-2 py-1 text-xs transition-colors"
                style={{
                  background: active ? "oklch(50% 0.13 226 / 0.10)" : "transparent",
                  color: active ? "var(--ch-accent)" : "rgb(var(--ch-t2))",
                }}
                title={`${p.provider} / ${p.model_id || "default"}`}
              >
                {p.name}
                <span className="text-ch-t4 ml-1 font-mono text-[10px]">
                  {p.provider}
                </span>
              </button>
            );
          })}
        </div>
      )}

      <button
        onClick={apply}
        disabled={!selPresetId || selPresetId === currentPresetId}
        className="w-full text-white text-xs font-medium py-1.5 rounded-md transition-opacity disabled:opacity-30 disabled:cursor-not-allowed"
        style={{ background: "var(--ch-accent)" }}
      >
        適用
      </button>
    </div>
  );
}
