/**
 * チャットヘッダーのモデル切り替えメニュー。
 *
 * フラットなモデル一覧（"{char}@{preset}" 形式）をキャラクター・プリセットの
 * 2 段選択に分けて表示する。NewSessionPicker の 1on1 設定と同じ操作感。
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { Model } from "../api";
import { charNameOf, presetNameOf } from "../api";
import { CharacterAvatar } from "./ChatBubbles";

interface Props {
  /** 利用可能なモデル一覧。 */
  models: Model[];
  /** 現在選択中のモデルID（"{char}@{preset}" 形式）。 */
  currentModelId: string;
  /** モデル確定時のコールバック。 */
  onApply: (modelId: string) => void;
  /** メニューを閉じるコールバック。 */
  onClose: () => void;
}

/** ヘッダー用のキャラクター/プリセット選択メニュー。 */
export default function CharPresetMenu({ models, currentModelId, onApply, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  /** モデル一覧をキャラクター名 → プリセット名配列にグルーピングする。 */
  const charMap = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const m of models) {
      const c = charNameOf(m.id);
      const p = presetNameOf(m.id);
      if (!map.has(c)) map.set(c, []);
      map.get(c)!.push(p);
    }
    return map;
  }, [models]);
  const charNames = useMemo(() => [...charMap.keys()], [charMap]);

  const [selChar, setSelChar] = useState(charNameOf(currentModelId));
  const [selPreset, setSelPreset] = useState(presetNameOf(currentModelId));

  const availablePresets = charMap.get(selChar) ?? [];

  // キャラ変更でプリセットが候補外になったら先頭へ寄せる。
  useEffect(() => {
    if (availablePresets.length > 0 && !availablePresets.includes(selPreset)) {
      setSelPreset(availablePresets[0]);
    }
  }, [selChar, availablePresets, selPreset]);

  // メニュー外クリック・Esc で閉じる。
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

  /** 選択中のキャラ・プリセットを確定する。 */
  const apply = () => {
    if (!selChar || !selPreset) return;
    onApply(`${selChar}@${selPreset}`);
    onClose();
  };

  return (
    <div
      ref={ref}
      className="absolute left-0 top-[calc(100%+6px)] z-50 bg-ch-bg rounded-xl p-3"
      style={{ border: "1px solid var(--ch-sep2)", boxShadow: "var(--ch-shadow)", width: 232 }}
    >
      {/* キャラクター選択 */}
      <div className="text-[10px] text-ch-t3 font-mono mb-1.5" style={{ letterSpacing: "0.06em" }}>
        CHARACTER
      </div>
      <div className="flex gap-1.5 flex-wrap mb-3">
        {charNames.map((c) => {
          const active = selChar === c;
          return (
            <button
              key={c}
              onClick={() => setSelChar(c)}
              className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors"
              style={{
                border: `1px solid ${active ? "var(--ch-accent)" : "var(--ch-sep2)"}`,
                background: active ? "oklch(50% 0.13 226 / 0.08)" : "transparent",
                color: active ? "var(--ch-accent)" : "rgb(var(--ch-t2))",
              }}
            >
              <CharacterAvatar characterName={c} size={16} />
              {c}
            </button>
          );
        })}
      </div>

      {/* プリセット選択 */}
      <div className="text-[10px] text-ch-t3 font-mono mb-1.5" style={{ letterSpacing: "0.06em" }}>
        PRESET <span className="opacity-60">· {availablePresets.length}</span>
      </div>
      <div className="flex flex-col gap-0.5 mb-3 max-h-44 overflow-y-auto">
        {availablePresets.map((p) => {
          const active = selPreset === p;
          return (
            <button
              key={p}
              onClick={() => setSelPreset(p)}
              className="text-left rounded-md px-2 py-1 text-xs transition-colors"
              style={{
                background: active ? "oklch(50% 0.13 226 / 0.10)" : "transparent",
                color: active ? "var(--ch-accent)" : "rgb(var(--ch-t2))",
              }}
            >
              {p}
            </button>
          );
        })}
      </div>

      <button
        onClick={apply}
        className="w-full text-white text-xs font-medium py-1.5 rounded-md transition-opacity"
        style={{ background: "var(--ch-accent)" }}
      >
        適用
      </button>
    </div>
  );
}
