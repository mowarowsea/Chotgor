/**
 * InlineEditor — バブルをその場で書き換えるための編集フォーム。
 */
import React, { useEffect, useState } from "react";

/**
 * バブル内インライン編集フォーム（textarea + 注記 + キャンセル/確定）。
 *
 * ユーザ発話の編集（1on1・シナリオ）と GM 発話の手動書き換えで同じ操作感を共有する。
 * Ctrl+Enter で確定、Esc でキャンセル（下部の MessageInput と同じキーバインド）。
 *
 * 本文は内部 state で持ち、`value` がサーバ最新値に入れ替わったら追従する
 * （再ストリーム完了でターンを取り直したときなど）。
 */
export function InlineEditor({
  value,
  tone,
  rows = 3,
  note,
  submitLabel = "送信",
  onSubmit,
  onCancel,
}: {
  /** 編集開始時の本文。外から変わったら（編集中でなければ）追従する。 */
  value: string;
  /** 配色。バブルの見た目に合わせる。 */
  tone: "user" | "character";
  /** textarea の初期行数。 */
  rows?: number;
  /** ボタン行の左に添える注記（「この発言以降は削除されます」等）。 */
  note?: string;
  /** 確定ボタンの文言。 */
  submitLabel?: string;
  /** 確定時のコールバック。空文字では呼ばれない。 */
  onSubmit: (newContent: string) => void;
  onCancel: () => void;
}) {
  const [text, setText] = useState(value);

  useEffect(() => {
    setText(value);
  }, [value]);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      submit();
    }
    if (e.key === "Escape") {
      onCancel();
    }
  };

  const isUser = tone === "user";

  return (
    <div className="flex flex-col gap-2 w-full">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        rows={rows}
        autoFocus
        className={`px-3.5 py-2 text-sm resize-y focus:outline-none w-full ${
          isUser ? "rounded-xl" : "rounded-lg"
        }`}
        style={
          isUser
            ? { background: "rgb(var(--ch-ub))", color: "rgb(var(--ch-ut))" }
            : { background: "rgb(var(--ch-s2))", color: "rgb(var(--ch-t1))" }
        }
      />
      <div className="flex gap-2 justify-end items-center">
        {note && <p className="text-ch-t4 text-[10px] mr-auto">{note}</p>}
        <button
          onClick={onCancel}
          className="text-ch-t3 hover:text-ch-t2 text-xs px-3 py-1.5 rounded transition-colors"
        >
          キャンセル
        </button>
        <button
          onClick={submit}
          disabled={!text.trim()}
          className="text-white text-xs px-3 py-1.5 rounded font-medium transition-colors disabled:opacity-30"
          style={{ background: "var(--ch-accent)" }}
        >
          {submitLabel}
        </button>
      </div>
    </div>
  );
}
