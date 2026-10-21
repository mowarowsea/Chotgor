/** シナリオチャットビュー共通のヘルパー。 */

import type { ScenarioTurn } from "../../api";

/**
 * 直近の送信（1 ストリームリクエスト）の起点ターンを返す。`turns` は 1 件以上あること。
 *
 * 末尾から同じ `generation_id` が続く範囲を 1 送信とみなし、その先頭を起点にする。
 * 直前がユーザ発話ならそちらを起点にする（ユーザ発話が generation に含まれない場合への備え）。
 * `generation_id` を持たない古いターンは、最後のユーザ発話（無ければ先頭）で代用する。
 */
export function findLatestSendStart(turns: ScenarioTurn[]): ScenarioTurn {
  const lastIdx = turns.length - 1;
  const gen = turns[lastIdx].generation_id;
  if (!gen) {
    for (let i = lastIdx; i >= 0; i--) {
      if (turns[i].speaker_type === "user") return turns[i];
    }
    return turns[0];
  }
  let start = lastIdx;
  while (start > 0 && turns[start - 1].generation_id === gen) start--;
  if (turns[start].speaker_type !== "user" && turns[start - 1]?.speaker_type === "user") {
    start--;
  }
  return turns[start];
}

/** 文字列の末尾空白・改行を取り除く（表示時のノイズ除去）。 */
export function trimEnd(s: string): string {
  return s.replace(/\s+$/u, "");
}

/** アバター用のプレースホルダー文字（名前の頭文字）。 */
export function avatarInitial(name: string): string {
  return name?.[0] ?? "?";
}
