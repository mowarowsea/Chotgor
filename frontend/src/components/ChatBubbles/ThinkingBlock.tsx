/**
 * ThinkingBlock — 思考ブロック・想起記憶の折りたたみ表示。
 * 記憶行（[category] … (score: X)）とワーキングメモリスレッド行の
 * パース＆色付けヘルパーを内包する。
 */
import { useCallback, useState } from "react";

import { translateText } from "../../api";

/** 記憶カテゴリの背景色・文字色マップ。 */
const CATEGORY_COLORS: Record<string, { bg: string; text: string }> = {
  identity:   { bg: "rgba(50,90,160,0.14)",  text: "#6888cc" },
  user:       { bg: "rgba(120,60,160,0.14)",  text: "#a878c8" },
  semantic:   { bg: "rgba(55,130,90,0.14)",   text: "#6aa882" },
  contextual: { bg: "rgba(140,105,50,0.14)",  text: "#c4a050" },
};

/** `[category] content  (score: X.XX)` 行を解析して色付きレンダリングする。 */
function ReasoningLine({ line }: { line: string }) {
  const match = /^\[([\w_]+)\] (.*?)\s+\(score: [\d.]+\)$/.exec(line);
  if (match) {
    const category = match[1] as string;
    const text = match[2];
    const colors = CATEGORY_COLORS[category];
    return (
      <div
        className="flex items-start gap-1.5 rounded px-2 py-0.5 my-0.5 text-xs"
        style={colors ? { background: colors.bg, color: colors.text } : { color: "#505050" }}
      >
        <span className="shrink-0 font-medium">[{category}]</span>
        <span>{text}</span>
      </div>
    );
  }
  return <div className="text-ch-t3 whitespace-pre-wrap text-xs">{line}</div>;
}

/** `[category] content  (score: X.XX)` 形式かどうかを判定する。 */
function isMemoryLine(line: string): boolean {
  return /^\[[\w_]+\] .+\(score: [\d.]+\)$/.test(line);
}

/** 想起したワーキングメモリスレッド行の先頭マーカー。 */
const THREAD_LINE_PREFIX = "⟦thread⟧ ";

function isThreadLine(line: string): boolean {
  return line.startsWith(THREAD_LINE_PREFIX);
}

/**
 * 想起したワーキングメモリスレッド1行を描画する。
 * バックエンドの ``⟦thread⟧ [type] summary 〈atmosphere_tag〉 → post`` 形式をパースする。
 */
function ThreadLine({ line }: { line: string }) {
  const match = /^⟦thread⟧ \[([\w_]+)\] (.*)$/.exec(line);
  if (!match) {
    return <div className="text-ch-t3 whitespace-pre-wrap text-xs">{line}</div>;
  }
  const type = match[1] as string;
  const rest = match[2];
  return (
    <div
      className="flex items-start gap-1.5 rounded px-2 py-0.5 my-0.5 text-xs"
      style={{ background: "var(--ch-sep)", color: "#b8bcc4" }}
    >
      <span className="shrink-0 font-medium">[{type}]</span>
      <span>{rest}</span>
    </div>
  );
}

/**
 * 思考ブロック・想起記憶を折りたたみ表示するコンポーネント。
 * ストリーミング中は自動展開する。
 * 展開状態かつ非ストリーミング時に翻訳ボタンを表示する。
 */
export function ThinkingBlock({
  content,
  streaming = false,
}: {
  content: string;
  streaming?: boolean;
}) {
  const [expanded, setExpanded] = useState(streaming);
  const [translation, setTranslation] = useState<string | null>(null);
  const [translating, setTranslating] = useState(false);
  const [translateError, setTranslateError] = useState<string | null>(null);

  const handleTranslate = useCallback(async () => {
    setTranslating(true);
    setTranslateError(null);
    try {
      const result = await translateText(content);
      setTranslation(result);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("[ThinkingBlock] 翻訳エラー:", e);
      setTranslateError(msg);
    } finally {
      setTranslating(false);
    }
  }, [content]);

  const lines = content.split("\n").filter((l) => l !== "");
  const memoryLines = lines.filter(isMemoryLine);
  const threadLines = lines.filter(isThreadLine);
  const sketchLines = lines.filter((l) => !isMemoryLine(l) && !isThreadLine(l));

  return (
    <div className="rounded-lg overflow-hidden text-xs mb-1" style={{ border: "1px solid var(--ch-sep)" }}>
      <div className="flex items-center">
        <button
          className="flex-1 flex items-center gap-1.5 px-3 py-1.5 text-ch-t3 hover:text-ch-t2 transition-colors text-left"
          onClick={() => setExpanded((e) => !e)}
        >
          <span className="text-[9px] opacity-50">{expanded ? "▼" : "▶"}</span>
          <span className="tracking-wide">想起した記憶・スケッチ</span>
          {streaming && <span className="animate-pulse ml-1 text-ch-accent-t text-[10px]">●</span>}
        </button>
        {expanded && !streaming && (
          <button
            className="px-2 py-1 mr-1 text-[10px] text-ch-t4 hover:text-ch-t2 transition-colors shrink-0 rounded"
            style={{ border: "1px solid var(--ch-sep)" }}
            onClick={handleTranslate}
            disabled={translating}
            title="日本語に翻訳"
          >
            {translating ? "…" : translation ? "再翻訳" : "翻訳"}
          </button>
        )}
      </div>
      {expanded && (
        <div className="px-3 py-2 font-mono leading-relaxed" style={{ borderTop: "1px solid var(--ch-sep)" }}>
          {memoryLines.length > 0 && (
            <>
              <div className="flex items-center gap-2 mb-1.5">
                <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
                <span className="text-ch-t4 shrink-0 text-[10px]">想起した記憶</span>
                <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
              </div>
              <div className="mb-1.5">
                {memoryLines.map((line, i) => (
                  <ReasoningLine key={i} line={line} />
                ))}
              </div>
            </>
          )}
          {threadLines.length > 0 && (
            <>
              <div className="flex items-center gap-2 my-1.5">
                <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
                <span className="text-ch-t4 shrink-0 text-[10px]">想起したスレッド</span>
                <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
              </div>
              <div className="mb-1.5">
                {threadLines.map((line, i) => (
                  <ThreadLine key={i} line={line} />
                ))}
              </div>
            </>
          )}
          {sketchLines.length > 0 && (
            <>
              <div className="flex items-center gap-2 my-1.5">
                <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
                <span className="text-ch-t4 shrink-0 text-[10px]">スケッチ</span>
                <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
              </div>
              <div className="text-ch-t3 whitespace-pre-wrap">
                {sketchLines.join("\n")}
              </div>
            </>
          )}
          {translateError && (
            <div className="mt-2 text-[10px]" style={{ color: "#c87070" }}>
              {translateError}
            </div>
          )}
          {translation && (
            <>
              <div className="flex items-center gap-2 mt-2 mb-1.5">
                <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
                <span className="text-ch-t4 shrink-0 text-[10px]">翻訳</span>
                <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
              </div>
              <div className="text-ch-t3 whitespace-pre-wrap">
                {translation}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
