/**
 * チャットバブル共通コンポーネント。
 * 1on1チャット・グループチャット両方で使用する表示・インタラクション部品を集約する。
 *
 * デザイン方針:
 *   キャラクターメッセージ — バブルボックスなし。アバター+名前ヘッダーの下に
 *                            インデントされたテキストを流す「ドキュメントスタイル」。
 *   ユーザーメッセージ     — 右寄せ、ニュートラルなフラット背景。
 *   ボーダー/サーフェス    — すべてニュートラルグレー。緑はキャラ名・アクセントのみ。
 */
import { useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

// ---------------------------------------------------------------------------
// ThinkingBlock
// ---------------------------------------------------------------------------

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

/**
 * 思考ブロック・想起記憶を折りたたみ表示するコンポーネント。
 * ストリーミング中は自動展開する。
 */
export function ThinkingBlock({
  content,
  streaming = false,
}: {
  content: string;
  streaming?: boolean;
}) {
  const [expanded, setExpanded] = useState(streaming);

  const lines = content.split("\n").filter((l) => l !== "");
  const memoryLines = lines.filter(isMemoryLine);
  const sketchLines = lines.filter((l) => !isMemoryLine(l));

  return (
    <div className="rounded-lg overflow-hidden text-xs mb-1" style={{ border: "1px solid rgba(255,255,255,0.07)" }}>
      <button
        className="w-full flex items-center gap-1.5 px-3 py-1.5 text-ch-t3 hover:text-ch-t2 transition-colors text-left"
        onClick={() => setExpanded((e) => !e)}
      >
        <span className="text-[9px] opacity-50">{expanded ? "▼" : "▶"}</span>
        <span className="tracking-wide">想起した記憶・スケッチ</span>
        {streaming && <span className="animate-pulse ml-1 text-ch-accent-t text-[10px]">●</span>}
      </button>
      {expanded && (
        <div className="px-3 py-2 font-mono leading-relaxed" style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
          {memoryLines.length > 0 && (
            <>
              <div className="flex items-center gap-2 mb-1.5">
                <hr className="flex-1" style={{ borderColor: "rgba(255,255,255,0.07)" }} />
                <span className="text-ch-t4 shrink-0 text-[10px]">想起した記憶</span>
                <hr className="flex-1" style={{ borderColor: "rgba(255,255,255,0.07)" }} />
              </div>
              <div className="mb-1.5">
                {memoryLines.map((line, i) => (
                  <ReasoningLine key={i} line={line} />
                ))}
              </div>
            </>
          )}
          {sketchLines.length > 0 && (
            <>
              <div className="flex items-center gap-2 my-1.5">
                <hr className="flex-1" style={{ borderColor: "rgba(255,255,255,0.07)" }} />
                <span className="text-ch-t4 shrink-0 text-[10px]">スケッチ</span>
                <hr className="flex-1" style={{ borderColor: "rgba(255,255,255,0.07)" }} />
              </div>
              <div className="text-ch-t3 whitespace-pre-wrap">
                {sketchLines.join("\n")}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// CopyButton
// ---------------------------------------------------------------------------

/** テキストをクリップボードにコピーし、完了時に一時的にチェックマークを表示するボタン。 */
function CopyButton({ text, className = "" }: { text: string; className?: string }) {
  const [done, setDone] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const el = document.createElement("textarea");
      el.value = text;
      el.style.cssText = "position:fixed;opacity:0;pointer-events:none";
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
    }
    setDone(true);
    setTimeout(() => setDone(false), 1500);
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      title="コピー"
      className={`opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-ch-t3 hover:text-ch-t2 transition-all p-1 rounded shrink-0 ${className}`}
    >
      {done ? (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" width={12} height={12}>
          <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={12} height={12}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0 0 13.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 0 1-.75.75H9a.75.75 0 0 1-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 0 1-2.25 2.25H6.75A2.25 2.25 0 0 1 4.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 0 1 1.927-.184" />
        </svg>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// CharacterAvatar
// ---------------------------------------------------------------------------

/**
 * キャラクターアバター。
 * imageUrl が指定されている場合は画像を表示し、未指定またはロード失敗時はイニシャルを表示する。
 */
export function CharacterAvatar({
  characterName,
  imageUrl,
  bgClass = "bg-ch-s2",
}: {
  characterName: string;
  imageUrl?: string;
  bgClass?: string;
}) {
  const [imgFailed, setImgFailed] = useState(false);
  return (
    <div
      className={`w-[50px] h-[50px] rounded-full ${bgClass} flex items-center justify-center text-xl font-semibold shrink-0 overflow-hidden`}
      style={{ border: "1px solid rgba(255,255,255,0.10)" }}
    >
      {imageUrl && !imgFailed ? (
        <img
          src={imageUrl}
          alt=""
          className="w-full h-full object-cover"
          onError={() => setImgFailed(true)}
        />
      ) : (
        <span className="text-ch-t2">{characterName.charAt(0)}</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// CharacterBubble
// ---------------------------------------------------------------------------

/**
 * キャラクターのチャットメッセージ（ドキュメントスタイル）。
 * バブルボックスなし。アバター+名前ヘッダーの下にインデントされたテキストを流す。
 */
export function CharacterBubble({
  characterName,
  content,
  reasoning,
  avatarBg = "bg-ch-s2",
  nameColor = "text-ch-accent-t",
  sending = false,
  onRegenerate,
  imageUrl,
}: {
  characterName: string;
  content: string;
  reasoning?: string;
  avatarBg?: string;
  nameColor?: string;
  sending?: boolean;
  onRegenerate?: () => void;
  imageUrl?: string;
}) {
  return (
    <div className="group" data-testid="character-bubble">
      {/* ヘッダー行: アバター + キャラクター名 */}
      <div className="flex items-center gap-3 mb-2">
        <CharacterAvatar characterName={characterName} imageUrl={imageUrl} bgClass={avatarBg} />
        <span className={`text-xs font-medium ${nameColor}`}>{characterName}</span>
      </div>

      {/* コンテンツ: gap12px インデント */}
      <div className="pl-[12px] space-y-1">
        {reasoning && <ThinkingBlock content={reasoning} />}
        <div className="text-ch-t1 text-sm leading-relaxed">
          <MarkdownContent content={content} />
        </div>
        {!sending && (
          <div className="flex items-center gap-0.5 -ml-1 mt-0.5">
            <CopyButton text={content} />
            {onRegenerate && (
              <button
                onClick={onRegenerate}
                title="再生成"
                className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-ch-t3 hover:text-ch-t2 text-xs transition-all p-1 rounded"
              >
                ↺
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// UserBubble
// ---------------------------------------------------------------------------

/**
 * ユーザーのチャットメッセージ。
 * 右寄せ・ニュートラルフラットデザイン。インライン編集フォームを内包する。
 */
export function UserBubble({
  content,
  userName,
  images,
  sending = false,
  onEdit,
}: {
  content: string;
  userName: string;
  images?: string[];
  sending?: boolean;
  onEdit?: (newContent: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(content);

  const handleEditSubmit = () => {
    const text = editText.trim();
    if (!text) return;
    setEditing(false);
    onEdit?.(text);
  };

  const handleEditKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && e.shiftKey) {
      e.preventDefault();
      handleEditSubmit();
    }
    if (e.key === "Escape") {
      setEditing(false);
      setEditText(content);
    }
  };

  return (
    <div className="group flex flex-col items-end gap-0.5">
      {/* ユーザー名ラベル */}
      <span className="text-[11px] text-ch-t4 pr-1">{userName}</span>

      {/* 添付画像 */}
      {images && images.length > 0 && <ImageGrid imageIds={images} />}

      {editing ? (
        <div className="flex flex-col gap-2 w-full max-w-[80%]">
          <textarea
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            onKeyDown={handleEditKeyDown}
            rows={3}
            autoFocus
            className="bg-ch-s2 text-ch-t1 rounded-xl px-4 py-2.5 text-sm resize-none focus:outline-none w-full"
            style={{ border: "1px solid rgba(255,255,255,0.14)" }}
          />
          <div className="flex gap-2 justify-end">
            <button
              onClick={() => { setEditing(false); setEditText(content); }}
              className="text-ch-t3 hover:text-ch-t2 text-xs px-3 py-1.5 rounded transition-colors"
            >
              キャンセル
            </button>
            <button
              onClick={handleEditSubmit}
              disabled={!editText.trim()}
              className="text-ch-accent-t bg-ch-accent-dim text-xs px-3 py-1.5 rounded transition-colors disabled:opacity-30"
              style={{ border: "1px solid rgba(77,140,103,0.30)" }}
            >
              送信
            </button>
          </div>
        </div>
      ) : (
        <div className="flex items-end gap-1 flex-row-reverse min-w-0 max-w-[80%]">
          <div
            className="rounded-xl px-4 py-2 text-ch-t1 text-sm overflow-hidden"
            style={{
              background: "rgba(22,22,22,0.95)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <MarkdownContent content={content} />
          </div>
          {!sending && (
            <div className="flex items-center gap-0.5">
              <CopyButton text={content} />
              {onEdit && (
                <button
                  onClick={() => setEditing(true)}
                  title="編集"
                  className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-ch-t3 hover:text-ch-t2 transition-all p-1 rounded shrink-0"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={12} height={12}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10" />
                  </svg>
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ImageGrid / ImageModal
// ---------------------------------------------------------------------------

/**
 * 添付画像IDリストをサムネイルグリッドで表示するコンポーネント。
 */
export function ImageGrid({ imageIds }: { imageIds: string[] }) {
  const [modalSrc, setModalSrc] = useState<string | null>(null);

  return (
    <>
      <div className="flex flex-wrap gap-1.5 justify-end mb-1">
        {imageIds.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setModalSrc(`/api/chat/images/${id}`)}
            className="block rounded-lg overflow-hidden transition-opacity hover:opacity-80"
            style={{ border: "1px solid rgba(255,255,255,0.10)" }}
          >
            <img
              src={`/api/chat/images/${id}`}
              alt="添付画像"
              className="w-20 h-20 object-cover"
            />
          </button>
        ))}
      </div>
      {modalSrc && (
        <ImageModal src={modalSrc} onClose={() => setModalSrc(null)} />
      )}
    </>
  );
}

/** 画像フルサイズ表示モーダル。 */
export function ImageModal({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
      style={{ backdropFilter: "blur(12px)" }}
      onClick={onClose}
    >
      <button
        onClick={onClose}
        className="absolute top-5 right-5 text-ch-t2 hover:text-ch-t1 text-xl leading-none"
      >
        ✕
      </button>
      <img
        src={src}
        alt="フルサイズ画像"
        className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg"
        style={{ boxShadow: "0 0 60px rgba(0,0,0,0.8)" }}
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// CodeBlock
// ---------------------------------------------------------------------------

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
          style={{ background: "rgba(15,15,15,0.85)" }}
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

/** Markdownテキストをレンダリングするコンポーネント。 */
export function MarkdownContent({ content }: { content: string }) {
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
                <div className="bg-ch-bg rounded-lg p-3 overflow-x-auto" style={{ border: "1px solid rgba(255,255,255,0.08)" }}>
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
          return <strong className="font-semibold text-ch-t1">{children}</strong>;
        },
        em({ children }) {
          return <em className="italic">{children}</em>;
        },
        h1({ children }) { return <h1 className="text-base font-semibold mt-3 mb-1 text-ch-t1">{children}</h1>; },
        h2({ children }) { return <h2 className="text-sm font-semibold mt-3 mb-1 text-ch-t1">{children}</h2>; },
        h3({ children }) { return <h3 className="text-sm font-medium mt-2 mb-1 text-ch-t1">{children}</h3>; },
        ul({ children }) { return <ul className="list-disc list-inside mb-2 space-y-0.5 text-ch-t1">{children}</ul>; },
        ol({ children }) { return <ol className="list-decimal list-inside mb-2 space-y-0.5 text-ch-t1">{children}</ol>; },
        hr() { return <hr className="my-3" style={{ borderColor: "rgba(255,255,255,0.10)" }} />; },
        blockquote({ children }) {
          return (
            <blockquote className="pl-3 text-ch-t2 my-2 italic" style={{ borderLeft: "2px solid rgba(255,255,255,0.15)" }}>
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
