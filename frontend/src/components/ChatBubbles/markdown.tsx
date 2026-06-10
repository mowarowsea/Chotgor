/**
 * Markdown レンダリング — CodeBlock（コピー付き）と memo 化 MarkdownContent。
 */
import React, { useCallback, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

/** コードブロック表示コンポーネント。右上にコピーボタンを重ねて表示する。 */
function CodeBlock({
  language,
  codeText,
  children,
}: {
  language?: string;
  codeText: string;
  children: React.ReactNode;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(codeText);
    } catch {
      const el = document.createElement("textarea");
      el.value = codeText;
      el.style.cssText = "position:fixed;opacity:0;pointer-events:none";
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [codeText]);

  return (
    <div className="relative group/code my-2 max-w-full overflow-x-auto">
      <div className="absolute top-2 right-2 flex items-center gap-1.5 z-10">
        {language && (
          <span className="text-[10px] text-ch-t3 font-mono select-none">{language}</span>
        )}
        <button
          onClick={handleCopy}
          title="コピー"
          className="opacity-0 group-hover/code:opacity-100 transition-opacity text-ch-t3 hover:text-ch-t2 rounded p-1"
          style={{ background: "rgb(var(--ch-s1))" }}
        >
          {copied ? (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" width={12} height={12}>
              <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={12} height={12}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0 0 13.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 0 1-.75.75H9a.75.75 0 0 1-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 0 1-2.25 2.25H6.75A2.25 2.25 0 0 1 4.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 0 1 1.927-.184" />
            </svg>
          )}
        </button>
      </div>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MarkdownContent
// ---------------------------------------------------------------------------

/** Markdownテキストをレンダリングするコンポーネント。
 *
 * 1on1 / シナリオ / グループチャットすべて同一スタイル。バブル間で見た目を
 * 揃えるため、種別ごとのバリアントは持たない（過去にあった variant="scenario" は
 * 廃止済み — `*行動描写*` の見た目は 1on1 と同じ italic に集約）。
 *
 * パフォーマンス: 末尾で `React.memo` 化されている（同名 const で再エクスポート）。
 * content prop が同一なら再レンダリングをスキップする — シンタックスハイライト・
 * GFM テーブル等のレンダリングコストを抑え、ストリーミング中の不要描画を削減する。
 */
function MarkdownContentImpl({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkBreaks]}
      components={{
        code({ className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          const rawCode = String(children);
          const isBlock = rawCode.endsWith("\n");
          const codeText = rawCode.replace(/\n$/, "");

          if (match) {
            return (
              <CodeBlock language={match[1]} codeText={codeText}>
                <SyntaxHighlighter
                  style={oneDark}
                  language={match[1]}
                  PreTag="div"
                  customStyle={{ borderRadius: "0.5rem", fontSize: "0.78rem", margin: 0, overflowX: "auto" }}
                >
                  {codeText}
                </SyntaxHighlighter>
              </CodeBlock>
            );
          }

          if (isBlock) {
            return (
              <CodeBlock codeText={codeText}>
                <div className="bg-ch-bg rounded-lg p-3 overflow-x-auto" style={{ border: "1px solid var(--ch-sep)" }}>
                  <code className="font-mono text-[0.78rem] text-ch-t1 whitespace-pre">{codeText}</code>
                </div>
              </CodeBlock>
            );
          }

          return (
            <code className="bg-ch-s3 text-ch-t2 rounded px-1 py-0.5 font-mono text-[0.83em]" {...props}>
              {children}
            </code>
          );
        },
        p({ children }) {
          return <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>;
        },
        strong({ children }) {
          // 色は指定せず親から継承する。キャラバブル（text-ch-t1）でも
          // ユーザバブル（紺地・var(--ut)）でも正しい文字色になる。
          return <strong className="font-semibold">{children}</strong>;
        },
        em({ children }) {
          // 斜体は行動描写・強調として使われるので青みグレーの emphasis 色を当てる。
          // PC とモバイルで明度を変える（モバイルは少し濃く）。
          return (
            <em className="italic mx-2 sm:text-ch-t-emphasis text-ch-t-emphasis-mobile">
              {children}
            </em>
          );
        },
        h1({ children }) { return <h1 className="text-base font-semibold mt-3 mb-1">{children}</h1>; },
        h2({ children }) { return <h2 className="text-sm font-semibold mt-3 mb-1">{children}</h2>; },
        h3({ children }) { return <h3 className="text-sm font-medium mt-2 mb-1">{children}</h3>; },
        ul({ children }) { return <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>; },
        ol({ children }) { return <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>; },
        hr() { return <hr className="my-3" style={{ borderColor: "var(--ch-sep)" }} />; },
        blockquote({ children }) {
          return (
            <blockquote className="pl-3 text-ch-t2 my-2 italic" style={{ borderLeft: "2px solid var(--ch-sep2)" }}>
              {children}
            </blockquote>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

/**
 * MarkdownContent の memo 化版（公開エクスポート）。
 *
 * 親が再レンダしても、content prop が同じなら ReactMarkdown ツリーを再構築しない。
 * シンタックスハイライト / コードブロック / GFM テーブル等が含まれる長いバブルでは
 * 描画コストが大きいため、ストリーミング中の不要再描画を抑えるのに効く。
 */
export const MarkdownContent = React.memo(MarkdownContentImpl, (prev, next) => {
  return prev.content === next.content;
});
