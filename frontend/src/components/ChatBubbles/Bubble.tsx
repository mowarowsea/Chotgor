/**
 * Bubble — 吹き出しの「箱」だけを担う共通プリミティブ。
 */
import React from "react";

import { bubbleClassFor } from "./colors";

/**
 * 吹き出しの外観（背景・角丸・枠線）を担う共通プリミティブ。
 * 1on1 / グループ / シナリオの全モードで同じ見た目を共有するために使う。
 *
 * - kind="user": 紺地・右下角を欠いた角丸
 * - kind="character": 左上角を欠いた角丸。colored 時はキャラクター別配色（cb0〜cb9）、
 *   それ以外はニュートラル面（bg-ch-s1）。
 */
export function Bubble({
  kind,
  characterName = "",
  colored = false,
  dashed = false,
  children,
}: {
  kind: "user" | "character";
  /** colored=true のときの配色決定に使うキャラクター名。 */
  characterName?: string;
  /** true でキャラクター別配色バブル（cb0〜cb9）。グループ/シナリオ向け。 */
  colored?: boolean;
  /** true で枠線を破線にする（シナリオの ephemeral NPC 用）。 */
  dashed?: boolean;
  children: React.ReactNode;
}) {
  if (kind === "user") {
    return (
      <div
        className="inline-block max-w-full px-3.5 py-2 text-sm leading-relaxed overflow-hidden break-words"
        style={{
          background: "rgb(var(--ch-ub))",
          color: "rgb(var(--ch-ut))",
          borderRadius: "14px 14px 4px 14px",
        }}
      >
        {children}
      </div>
    );
  }
  return (
    <div
      className={`inline-block max-w-full px-3.5 py-2.5 text-ch-t1 text-sm leading-relaxed break-words ${colored ? bubbleClassFor(characterName) : "bg-ch-s1"}`}
      style={{
        borderRadius: "4px 14px 14px 14px",
        border: dashed ? "1px dashed" : "1px solid",
        // colored 時は cb クラスの border-color を活かすため未指定にする。
        borderColor: colored ? undefined : "var(--ch-sep)",
      }}
    >
      {children}
    </div>
  );
}
