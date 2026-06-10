/**
 * シナリオチャットの NPC 表示部品 — 円形アバターと NPC 詳細ダイアログ。
 */
import { useEffect } from "react";

import type { ScenarioNpc } from "../../api";
import { charHue } from "../ChatBubbles";
import { avatarInitial } from "./helpers";

/**
 * アバター画像 or プレースホルダーを表示する小さな円形要素。
 *
 * onClick が渡されると button としてラップされ、ホバーで微かに浮き上がる
 * （NPC 詳細ダイアログを開く操作のヒント）。onClick が無ければただの装飾。
 */
export function Avatar(props: {
  name: string;
  src: string | null;
  size?: number;
  onClick?: () => void;
}) {
  const size = props.size ?? 28;
  const common = {
    width: size,
    height: size,
    borderRadius: "50%",
    flexShrink: 0,
  } as const;
  // 配色はキャラクター名から導出した色相を使い、1on1 チャットのアバターと揃える。
  const h = charHue(props.name);
  const inner = props.src ? (
    <img
      src={props.src}
      alt={props.name}
      style={{ ...common, objectFit: "cover" }}
      className="bg-ch-s2"
    />
  ) : (
    <div
      style={{
        ...common,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: size * 0.38,
        fontWeight: 600,
        background: `oklch(56% 0.12 ${h} / 0.15)`,
        color: `oklch(44% 0.14 ${h})`,
      }}
    >
      {avatarInitial(props.name)}
    </div>
  );

  if (!props.onClick) return inner;
  return (
    <button
      type="button"
      onClick={props.onClick}
      title={`${props.name} の詳細を表示`}
      className="rounded-full transition-transform hover:scale-105 hover:ring-1 hover:ring-ch-s3 focus:outline-none focus:ring-1 focus:ring-ch-accent"
      style={{ display: "inline-block", lineHeight: 0 }}
    >
      {inner}
    </button>
  );
}

/**
 * NPC 詳細ダイアログ。アバタークリックで開く。
 *
 * シンプル構成: 画像（大）/ 名前 / description（自由テキスト）。
 * 既知 NPC のみ呼ばれる想定（npc が null なら何も描画しない）。
 */
export function NpcDetailDialog({
  npc,
  onClose,
}: {
  npc: ScenarioNpc | null;
  onClose: () => void;
}) {
  // Esc で閉じる挙動。1on1 等の他のダイアログと操作感を合わせる。
  useEffect(() => {
    if (!npc) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [npc, onClose]);

  if (!npc) return null;
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center px-4"
      style={{ background: "var(--ch-overlay)" }}
      onClick={onClose}
    >
      <div
        className="bg-ch-bg rounded-xl w-full max-w-md overflow-hidden"
        style={{ border: "1px solid var(--ch-sep2)", boxShadow: "var(--ch-shadow)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 画像エリア（あれば大きめ） */}
        {npc.image_data ? (
          <div className="w-full bg-ch-s1 flex items-center justify-center">
            <img
              src={npc.image_data}
              alt={npc.name}
              className="max-h-72 w-full object-contain"
            />
          </div>
        ) : (
          <div className="w-full bg-ch-s1 flex items-center justify-center py-12">
            <div
              style={{ width: 96, height: 96, borderRadius: "50%", fontSize: 40 }}
              className="bg-ch-s2 text-ch-t2 flex items-center justify-center"
            >
              {avatarInitial(npc.name)}
            </div>
          </div>
        )}

        {/* 本文 */}
        <div className="px-5 py-4">
          <div className="flex items-start justify-between gap-3 mb-3">
            <h2 className="text-ch-t1 text-base font-medium break-words">
              {npc.name}
            </h2>
            <button
              onClick={onClose}
              className="text-ch-t3 hover:text-ch-t1 text-sm -mr-1 -mt-1 p-1 rounded"
              aria-label="閉じる"
              title="閉じる (Esc)"
            >
              ✕
            </button>
          </div>
          {npc.description ? (
            <p className="text-ch-t2 text-sm leading-relaxed whitespace-pre-wrap break-words">
              {npc.description}
            </p>
          ) : (
            <p className="text-ch-t4 text-sm italic">説明はありません</p>
          )}
        </div>
      </div>
    </div>
  );
}
