/**
 * ThinkingBlock — 想起記憶ブロックと思考(スケッチ)ブロックを別アコーディオンで表示する。
 * バックエンドは両者を1本の reasoning テキストとして送ってくるため、
 * 記憶行（[category] … (score: X)）とワーキングメモリスレッド行を判定して
 * 「想起した記憶」グループと「スケッチ」グループに仕分けたうえで、
 * それぞれ独立して開閉できる折りたたみとして描画する。
 */
import { useCallback, useState } from "react";
import type { ReactNode } from "react";

import { translateText } from "../../api";

/** 記憶カテゴリの背景色・文字色マップ。 */
const CATEGORY_COLORS: Record<string, { bg: string; text: string }> = {
  identity:   { bg: "rgba(50,90,160,0.14)",  text: "#6888cc" },
  user:       { bg: "rgba(120,60,160,0.14)",  text: "#a878c8" },
  semantic:   { bg: "rgba(55,130,90,0.14)",   text: "#6aa882" },
  contextual: { bg: "rgba(140,105,50,0.14)",  text: "#c4a050" },
};

/** `(origin_label? )?[category] content  (score: X.XX)` 行を解析して色付きレンダリングする。
 *
 * バックエンド format_recalled_memories は origin が usual/interlude のとき行頭に
 * 「[うつつでの記憶] 」「[TRPGでの記憶] 」を付ける。category は ASCII 固定なので
 * `[\w_]+` のままだが、origin ラベル部は日本語を含むので `[^\]]+` で吸収する。
 * 文字クラス `\w` は JS では既定 ASCII のみで、`[\w_]+` では日本語ラベルが
 * 捕捉できず行が「スケッチ」に流れる事故が起きていた（findings #3）。
 */
function ReasoningLine({ line }: { line: string }) {
  const match = /^(?:\[[^\]]+\] )?\[([\w_]+)\] (.*?)\s+\(score: [\d.]+\)$/.exec(line);
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

/** memory line 判定。`(score: X.XX)` で終わる行を一律 memory line とみなす（findings #3）。
 *
 * 旧実装は `/^\[[\w_]+\] .+\(score: [\d.]+\)$/` で行頭の `[category]` を見ていたが、
 * origin ラベル `[うつつでの記憶] [contextual] ...` のように先頭ブラケットが日本語を含むと
 * `\w` が ASCII 限定のため判定漏れし、行が「スケッチ」欄へ落ちていた。score 末尾だけで
 * 判定すればラベル有無を問わず memory line として拾える。
 */
function isMemoryLine(line: string): boolean {
  return /\(score: [\d.]+\)\s*$/.test(line);
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
  // origin ラベル（日本語含む）が type の前に挟まるパターンに対応（findings #3）。
  // `(?:\[[^\]]+\] )?` で `[うつつでの記憶] ` / `[TRPGでの記憶] ` を任意に飲み込む。
  const match = /^⟦thread⟧ (?:\[[^\]]+\] )?\[([\w_]+)\] (.*)$/.exec(line);
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
 * 折りたたみ可能な補助ブロックの共通シェル。
 * 「想起した記憶」ブロックと「スケッチ」ブロックはこれを土台に個別インスタンス化され、
 * 開閉状態・翻訳状態を互いに独立して持つ（これが別アコーディオン化の実体）。
 * ストリーミング中は自動展開する。展開状態かつ非ストリーミング時に翻訳ボタンを表示する。
 */
function CollapsibleAuxBlock({
  label,
  streaming = false,
  translateSource,
  children,
}: {
  label: string;
  streaming?: boolean;
  /** 翻訳ボタン押下時に送る原文。このブロックが担当する行のみを渡す。 */
  translateSource: string;
  children: ReactNode;
}) {
  const [expanded, setExpanded] = useState(streaming);
  const [translation, setTranslation] = useState<string | null>(null);
  const [translating, setTranslating] = useState(false);
  const [translateError, setTranslateError] = useState<string | null>(null);

  const handleTranslate = useCallback(async () => {
    setTranslating(true);
    setTranslateError(null);
    try {
      const result = await translateText(translateSource);
      setTranslation(result);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("[ThinkingBlock] 翻訳エラー:", e);
      setTranslateError(msg);
    } finally {
      setTranslating(false);
    }
  }, [translateSource]);

  return (
    <div className="ch-aux-bubble rounded-lg overflow-hidden text-xs mb-1" style={{ border: "1px solid var(--ch-sep)" }}>
      <div className="flex items-center">
        <button
          className="flex-1 flex items-center gap-1.5 px-3 py-1.5 text-ch-t3 hover:text-ch-t2 transition-colors text-left"
          onClick={() => setExpanded((e) => !e)}
        >
          <span className="text-[9px] opacity-50">{expanded ? "▼" : "▶"}</span>
          <span className="tracking-wide">{label}</span>
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
          {children}
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

/**
 * 想起記憶ブロックと思考(スケッチ)ブロックを別アコーディオンで表示するコンポーネント。
 * バックエンドから届く reasoning は1本の文字列だが、行の形（記憶行/スレッド行/それ以外）で
 * 「想起した記憶」グループと「スケッチ」グループに仕分け、それぞれ独立した
 * CollapsibleAuxBlock として描画する。どちらか一方しか無ければ他方は描画しない。
 */
export function ThinkingBlock({
  content,
  streaming = false,
}: {
  content: string;
  streaming?: boolean;
}) {
  const lines = content.split("\n").filter((l) => l !== "");
  const memoryLines = lines.filter(isMemoryLine);
  const threadLines = lines.filter(isThreadLine);
  const sketchLines = lines.filter((l) => !isMemoryLine(l) && !isThreadLine(l));

  const hasRecall = memoryLines.length > 0 || threadLines.length > 0;
  const hasSketch = sketchLines.length > 0;

  if (!hasRecall && !hasSketch) return null;

  return (
    <>
      {hasRecall && (
        <CollapsibleAuxBlock
          label="想起した記憶"
          streaming={streaming}
          translateSource={[...memoryLines, ...threadLines].join("\n")}
        >
          {memoryLines.length > 0 && (
            <div className="mb-1.5">
              {memoryLines.map((line, i) => (
                <ReasoningLine key={i} line={line} />
              ))}
            </div>
          )}
          {threadLines.length > 0 && (
            <>
              {memoryLines.length > 0 && (
                <div className="flex items-center gap-2 my-1.5">
                  <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
                  <span className="text-ch-t4 shrink-0 text-[10px]">想起したスレッド</span>
                  <hr className="flex-1" style={{ borderColor: "var(--ch-sep)" }} />
                </div>
              )}
              <div>
                {threadLines.map((line, i) => (
                  <ThreadLine key={i} line={line} />
                ))}
              </div>
            </>
          )}
        </CollapsibleAuxBlock>
      )}
      {hasSketch && (
        <CollapsibleAuxBlock
          label="スケッチ"
          streaming={streaming}
          translateSource={sketchLines.join("\n")}
        >
          <div className="text-ch-t3 whitespace-pre-wrap">
            {sketchLines.join("\n")}
          </div>
        </CollapsibleAuxBlock>
      )}
    </>
  );
}
