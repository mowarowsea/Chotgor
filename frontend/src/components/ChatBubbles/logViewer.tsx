/**
 * デバッグログ表示部品 — ToolCallRow / JSON ハイライト / Raw ログモーダル。
 * MessageActionBar のログ折りたたみから使用する（CHOTGOR_DEBUG=1 時）。
 */
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { fetchRawLog } from "../../api";
import type { LogTag, LogToolCall } from "../../api";

/** タグ種別 cls → Tailwindクラスのマッピング。 */
export const TAG_COLORS: Record<string, string> = {
  "tag-memory":     "bg-violet-950/60 text-violet-300",
  "tag-narrative":  "bg-blue-950/60 text-blue-300",
  "tag-drift":      "bg-amber-950/60 text-amber-300",
  "tag-switch":     "bg-teal-950/60 text-teal-300",
  "tag-recall":     "bg-rose-950/60 text-rose-300",
  "tag-anticipate": "bg-fuchsia-950/60 text-fuchsia-300",
  "tag-end":        "bg-ch-s3 text-ch-t2",
  "tag-unknown":    "bg-ch-s3 text-ch-t3",
};

/**
 * タグ1件の詳細表示（展開時の中身）。
 * ToolCallRow（過去ログ互換）と ToolTagRow（実行イベント由来）で共用する。
 * 実行イベント由来のタグは実行失敗時に error_message を併せて表示する。
 */
function TagDetail({ tag }: { tag: LogTag }) {
  return (
    <div className="ch-aux-bubble rounded px-2 py-1 text-[10px]" style={{ background: "var(--ch-sep)", border: "1px solid var(--ch-sep)" }}>
      <div className="text-ch-t3 font-mono mb-0.5">
        {tag.tag_name}
        {tag.status === "error" && <span className="text-red-400 ml-1.5">実行失敗</span>}
      </div>
      {Object.entries(tag.fields).map(([k, v]) => (
        <div key={k} className="flex gap-1.5 text-ch-t3">
          <span className="text-ch-t4 shrink-0">{k}:</span>
          <span className="text-ch-t2 break-all whitespace-pre-wrap">{v}</span>
        </div>
      ))}
      {tag.error_message && (
        <div className="text-red-400/80 break-all whitespace-pre-wrap mt-0.5">{tag.error_message}</div>
      )}
    </div>
  );
}

/**
 * 実行イベント（tool_call_events）由来のツールタグ一覧の表示。
 * 2026-06-11 のイベント記録方式移行後のログで使用する。
 * 過去ログ互換の ToolCallRow と異なり Request/Response ファイルを持たず、
 * タグバッジと展開詳細のみを表示する。
 */
export function ToolTagRow({ tags }: { tags: LogTag[] }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="ch-aux-bubble rounded px-2 py-1.5 text-[11px]" style={{ background: "var(--ch-sep)", border: "1px solid var(--ch-sep)" }}>
      <div className="flex items-center gap-1 flex-wrap">
        {tags.map((tag, i) => (
          <button
            key={i}
            onClick={() => setExpanded((e) => !e)}
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-opacity hover:opacity-80 ${TAG_COLORS[tag.meta.cls] ?? TAG_COLORS["tag-unknown"]}`}
          >
            {tag.meta.label}
            {tag.status === "error" && <span className="text-red-400 ml-0.5">⚠</span>}
          </button>
        ))}
      </div>
      {expanded && (
        <div className="mt-1.5 space-y-1">
          {tags.map((tag, i) => (
            <TagDetail key={i} tag={tag} />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * ToolCall 1件の表示。Request/Response ファイルリンクとタグバッジを表示する。
 *
 * ファイルの実取得は親（MessageActionBar）の RawLogModal が試行ごとの
 * フォルダID（dir_id）で行うため、本コンポーネントはファイル名だけを通知する。
 */
export function ToolCallRow({
  tc,
  onOpenRaw,
}: {
  tc: LogToolCall;
  onOpenRaw: (filename: string) => void;
}) {
  const [tagExpanded, setTagExpanded] = useState(false);

  return (
    <div className="ch-aux-bubble rounded px-2 py-1.5 text-[11px]" style={{ background: "var(--ch-sep)", border: "1px solid var(--ch-sep)" }}>
      {/* ヘッダー: feature / preset */}
      <div className="flex items-center gap-1.5 mb-1 text-ch-t3">
        {tc.feature && <span className="font-mono">{tc.feature}</span>}
        {tc.feature && tc.preset && <span className="opacity-40">/</span>}
        {tc.preset && <span className="font-mono">{tc.preset}</span>}
        {tc.request_file && !tc.response_file && (
          <span className="ml-auto text-red-400 text-[10px]">応答なし</span>
        )}
      </div>

      {/* ファイルリンク */}
      <div className="flex items-center gap-2 flex-wrap">
        {tc.request_file && (
          <button
            onClick={() => onOpenRaw(tc.request_file!)}
            className="text-ch-t4 hover:text-ch-t2 underline underline-offset-2 text-[10px] transition-colors"
          >
            Request
          </button>
        )}
        {tc.response_file && (
          <button
            onClick={() => onOpenRaw(tc.response_file!)}
            className="text-ch-t4 hover:text-ch-t2 underline underline-offset-2 text-[10px] transition-colors"
          >
            Response
          </button>
        )}

        {/* タグバッジ */}
        {tc.tags.length > 0 && (
          <div className="flex items-center gap-1 flex-wrap ml-auto">
            {tc.tags.map((tag, i) => (
              <button
                key={i}
                onClick={() => setTagExpanded((e) => !e)}
                className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-opacity hover:opacity-80 ${TAG_COLORS[tag.meta.cls] ?? TAG_COLORS["tag-unknown"]}`}
              >
                {tag.meta.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* タグ詳細（展開時） */}
      {tagExpanded && tc.tags.length > 0 && (
        <div className="mt-1.5 space-y-1">
          {tc.tags.map((tag, i) => (
            <TagDetail key={i} tag={tag} />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// HighlightedJson（JSON整形＋フィールドハイライト）
// ---------------------------------------------------------------------------

/** ハイライト対象のキー名セット（Claude CLI / Gemini 向けキーも含む）。 */
const HIGHLIGHT_KEYS = new Set(["text", "content", "thought", "system_prompt", "system_instruction", "conversation", "thinking", "result"]);
/** JSON の key-value 行にマッチする正規表現（indent=2 前提）。 */
const KV_LINE_RE = /^(\s*)"([^"]+)"(\s*:\s*)(.+)$/;

/**
 * 文字を1つずつ走査して JSON 文字列値内の生改行・CR を
 * JSON エスケープに置換する。大きなファイルでも O(n) で確実に動作する。
 */
function fixRawControlChars(text: string): string {
  const out: string[] = [];
  let inStr = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (!inStr) {
      out.push(ch);
      if (ch === '"') inStr = true;
    } else if (ch === "\\") {
      out.push(ch);
      out.push(text[++i] ?? "");
    } else if (ch === '"') {
      out.push(ch);
      inStr = false;
    } else if (ch === "\n") {
      out.push("\\n");
    } else if (ch === "\r") {
      out.push("\\r");
    } else {
      out.push(ch);
    }
  }
  return out.join("");
}

/**
 * JSON テキストのパースを試みる。
 * debug_logger が生成する不正エスケープや生改行を段階的に前処理して再試行する。
 * 通常の JSON パースが失敗した場合は NDJSON（複数行またがり対応）として解析する。
 */
function tryParseJson(text: string): unknown | null {
  try { return JSON.parse(text); } catch {}

  // \<LF> を \\n に置換
  const fixed1 = text.replace(/\\\n/g, "\\n");
  try { return JSON.parse(fixed1); } catch {}

  // 文字ベースの処理で生改行・CR を確実に置換（大ファイル対応）
  const fixed2 = fixRawControlChars(fixed1);
  try { return JSON.parse(fixed2); } catch {}

  // NDJSON（行またがり対応）として解析する
  const lines = fixed1.split("\n");
  const objects: unknown[] = [];
  const acc: string[] = [];
  for (const line of lines) {
    acc.push(line);
    try {
      objects.push(JSON.parse(fixRawControlChars(acc.join("\n"))));
      acc.length = 0;
    } catch { /* まだ不完全 */ }
  }
  if (objects.length > 0 && acc.every(l => !l.trim())) return objects;

  return null;
}

/**
 * JSON を整形しつつ、HIGHLIGHT_KEYS に含まれるキーの文字列値を
 * 緑マーカーでハイライトして表示するコンポーネント。
 * パース失敗時はプレーンテキストにフォールバックする。
 */
function HighlightedJson({ raw }: { raw: string }) {
  const preClass = "text-ch-t2 text-[11px] font-mono whitespace-pre-wrap break-all leading-relaxed";

  const parsed = tryParseJson(raw);
  if (!parsed) {
    return <pre className={preClass}>{raw}</pre>;
  }

  const pretty = JSON.stringify(parsed, null, 2);
  const lines = pretty.split("\n");

  return (
    <pre className={preClass}>
      {lines.map((line, i) => {
        const m = KV_LINE_RE.exec(line);
        if (m) {
          const [, indent, key, sep, valuePart] = m;
          if (HIGHLIGHT_KEYS.has(key)) {
            const trailing = valuePart.endsWith(",") ? "," : "";
            const stripped = trailing ? valuePart.slice(0, -1) : valuePart;
            // 文字列値（長さ 2 超、null 文字列でない）のみハイライト
            if (stripped.startsWith('"') && stripped !== '"null"' && stripped.length > 2) {
              // JSON.parse で内部のエスケープ（\n \t \" 等）を実際の文字に展開する
              let displayValue: string = stripped;
              try {
                const inner = JSON.parse(stripped) as unknown;
                if (typeof inner === "string") displayValue = inner;
              } catch { /* フォールバック: そのまま */ }
              return (
                <span key={i}>
                  {indent}
                  <span style={{ color: "var(--json-hl-key)" }}>"{key}"</span>
                  {sep}
                  <mark style={{ background: "var(--json-hl-bg)", color: "var(--json-hl-key)", borderRadius: "3px", padding: "0 2px", whiteSpace: "pre-wrap" }}>
                    {displayValue}
                  </mark>
                  {trailing}
                  {"\n"}
                </span>
              );
            }
          }
        }
        return <span key={i}>{line}{"\n"}</span>;
      })}
    </pre>
  );
}

// ---------------------------------------------------------------------------
// RawLogModal
// ---------------------------------------------------------------------------

/**
 * Raw ログファイルをオーバーレイモーダルで表示するコンポーネント。
 * JSON ファイルは整形＋フィールドハイライト表示する。
 */
export function RawLogModal({ messageId, filename, onClose }: { messageId: string; filename: string; onClose: () => void }) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // マウント時にファイル内容を取得する
  useEffect(() => {
    fetchRawLog(messageId, filename)
      .then(setContent)
      .catch((e) => setContent(`Error: ${String(e)}`))
      .finally(() => setLoading(false));
  }, [messageId, filename]);

  return createPortal(
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center p-4"
      style={{ background: "var(--ch-overlay)", backdropFilter: "blur(4px)" }}
      onClick={onClose}
    >
      <div
        className="bg-ch-bg rounded-xl w-full max-w-[90vw] max-h-[90vh] flex flex-col overflow-hidden"
        style={{ border: "1px solid var(--ch-sep2)", boxShadow: "var(--ch-shadow)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* ヘッダー */}
        <div className="flex items-center justify-between px-4 py-2.5 shrink-0" style={{ borderBottom: "1px solid var(--ch-sep)" }}>
          <span className="text-ch-t3 text-xs font-mono truncate">{filename}</span>
          <button onClick={onClose} className="text-ch-t3 hover:text-ch-t1 ml-4 text-sm">✕</button>
        </div>
        {/* コンテンツ */}
        <div className="overflow-y-auto flex-1 p-4">
          {loading ? (
            <div className="text-ch-t4 text-xs animate-pulse">読み込み中...</div>
          ) : (
            <HighlightedJson raw={content ?? ""} />
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}
