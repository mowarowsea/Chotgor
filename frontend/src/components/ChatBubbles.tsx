/**
 * チャットバブル共通コンポーネント。
 * 1on1チャット・グループチャット両方で使用する表示・インタラクション部品を集約する。
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

/**
 * 思考ブロック・想起記憶を折りたたみ表示するコンポーネント。
 * ストリーミング中は自動展開し、完了後は折りたたみ可能にする。
 */
export function ThinkingBlock({
  content,
  streaming = false,
}: {
  content: string;
  /** true のときはストリーミング中（自動展開・インジケーター表示）。 */
  streaming?: boolean;
}) {
  const [expanded, setExpanded] = useState(streaming);

  return (
    <div className="border border-zinc-700 rounded-xl overflow-hidden text-xs">
      <button
        className="w-full flex items-center gap-1.5 px-3 py-1.5 text-zinc-400 hover:bg-zinc-800/60 transition-colors text-left"
        onClick={() => setExpanded((e) => !e)}
      >
        <span className="text-[10px]">{expanded ? "▼" : "▶"}</span>
        <span>思考・想起した記憶</span>
        {streaming && <span className="animate-pulse ml-1 text-indigo-400">●</span>}
      </button>
      {expanded && (
        <div className="px-3 py-2 text-zinc-500 whitespace-pre-wrap font-mono border-t border-zinc-700/60 leading-relaxed">
          {content}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// CopyButton（内部共通）
// ---------------------------------------------------------------------------

/** テキストをクリップボードにコピーし、完了時に一時的にチェックマークを表示するボタン。 */
function CopyButton({ text, className = "" }: { text: string; className?: string }) {
  const [done, setDone] = useState(false);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setDone(true);
    setTimeout(() => setDone(false), 1500);
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      title="コピー"
      className={`opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-zinc-500 hover:text-zinc-300 transition-all p-1 rounded shrink-0 ${className}`}
    >
      {done ? (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" width={14} height={14}>
          <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={14} height={14}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0 0 13.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 0 1-.75.75H9a.75.75 0 0 1-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 0 1-2.25 2.25H6.75A2.25 2.25 0 0 1 4.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 0 1 1.927-.184" />
        </svg>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// CharacterBubble
// ---------------------------------------------------------------------------

/**
 * キャラクターのチャットバブル。
 * キャラクター名ラベル・reasoning・本文を表示し、オプションで再生成ボタンを表示する。
 * アバター背景色・名前ラベル色は呼び出し元で指定し、1on1/グループで使い分ける。
 */
export function CharacterBubble({
  characterName,
  content,
  reasoning,
  avatarBg = "bg-indigo-600",
  nameColor = "text-indigo-400",
  sending = false,
  onRegenerate,
}: {
  characterName: string;
  content: string;
  reasoning?: string;
  /** アバター背景色 Tailwind クラス（デフォルト: bg-indigo-600）。 */
  avatarBg?: string;
  /** キャラクター名ラベルの文字色 Tailwind クラス（デフォルト: text-indigo-400）。 */
  nameColor?: string;
  /** 送信処理中フラグ（true のとき再生成ボタンを非表示にする）。 */
  sending?: boolean;
  /** 再生成コールバック（省略時はボタン非表示）。 */
  onRegenerate?: () => void;
}) {
  return (
    <div className="group" data-testid="character-bubble">
      <div className="flex gap-3 items-start">
        <div className={`w-8 h-8 rounded-full ${avatarBg} flex items-center justify-center text-xs font-bold shrink-0`}>
          {characterName.charAt(0)}
        </div>
        <div className="max-w-[85%] sm:max-w-[70%] space-y-0.5">
          <p className={`text-xs font-medium ${nameColor} px-1`}>{characterName}</p>
          {reasoning && <ThinkingBlock content={reasoning} />}
          <div className="bg-zinc-800 rounded-2xl rounded-tl-sm px-4 py-2.5 text-zinc-100 text-sm">
            <MarkdownContent content={content} />
          </div>
        </div>
      </div>
      {/* アクションエリア（アバター幅分インデント、モバイルは常時・デスクトップはホバー時） */}
      {!sending && (
        <div className="pl-11 flex items-center gap-1">
          <CopyButton text={content} />
          {onRegenerate && (
            <button
              onClick={onRegenerate}
              title="再生成"
              className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-zinc-500 hover:text-zinc-300 text-xs transition-all px-2 py-1 rounded hover:bg-zinc-800"
            >
              ↺ 再生成
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// UserBubble
// ---------------------------------------------------------------------------

/**
 * ユーザーのチャットバブル。
 * 本文・添付画像を表示し、オプションでインライン編集フォームを表示する。
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
  /** 添付画像IDリスト（省略可）。 */
  images?: string[];
  /** 送信処理中フラグ（true のとき編集ボタンを非表示にする）。 */
  sending?: boolean;
  /** 編集確定コールバック（省略時はボタン非表示）。 */
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
    <div className="flex gap-3 items-start flex-row-reverse group">
      <div className="w-8 h-8 rounded-full bg-zinc-600 flex items-center justify-center text-xs font-bold shrink-0">
        {userName.charAt(0)}
      </div>
      <div className={`flex flex-col items-end gap-1 ${editing ? "w-full" : "max-w-[85%] sm:max-w-[70%]"}`}>
        {images && images.length > 0 && <ImageGrid imageIds={images} />}
        {editing ? (
          /* インライン編集フォーム */
          <div className="flex flex-col gap-2 w-full">
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              onKeyDown={handleEditKeyDown}
              rows={3}
              autoFocus
              className="bg-zinc-700 text-zinc-100 rounded-xl px-4 py-2.5 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-indigo-500 w-full"
            />
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => { setEditing(false); setEditText(content); }}
                className="text-zinc-400 hover:text-zinc-200 text-xs px-3 py-1.5 rounded-lg hover:bg-zinc-800 transition-colors"
              >
                キャンセル
              </button>
              <button
                onClick={handleEditSubmit}
                disabled={!editText.trim()}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs px-3 py-1.5 rounded-lg transition-colors"
              >
                送信
              </button>
            </div>
          </div>
        ) : (
          /* 通常表示 + ホバー時にコピー・編集ボタン */
          <div className="flex items-end gap-1 flex-row-reverse">
            <div className="bg-indigo-900 rounded-2xl rounded-tr-sm px-4 py-2.5 text-zinc-100 text-sm">
              <MarkdownContent content={content} />
            </div>
            {!sending && (
              <div className="flex items-center gap-0.5">
                <CopyButton text={content} />
                {onEdit && (
                  <button
                    onClick={() => setEditing(true)}
                    title="編集"
                    className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-zinc-500 hover:text-zinc-300 transition-all p-1 rounded shrink-0"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={14} height={14}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10" />
                    </svg>
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ImageGrid / ImageModal（1on1添付画像用）
// ---------------------------------------------------------------------------

/**
 * 添付画像IDリストをサムネイルグリッドで表示するコンポーネント。
 * クリックするとフルサイズモーダルを開く。
 */
export function ImageGrid({ imageIds }: { imageIds: string[] }) {
  const [modalSrc, setModalSrc] = useState<string | null>(null);

  return (
    <>
      <div className="flex flex-wrap gap-1.5 justify-end">
        {imageIds.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setModalSrc(`/api/chat/images/${id}`)}
            className="block rounded-lg overflow-hidden border border-zinc-700 hover:border-zinc-500 transition-colors"
          >
            <img
              src={`/api/chat/images/${id}`}
              alt="添付画像"
              className="w-24 h-24 object-cover"
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

/**
 * 画像フルサイズ表示モーダル。
 * 背景クリックまたは✕ボタンで閉じる。
 */
export function ImageModal({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center"
      onClick={onClose}
    >
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-white text-2xl leading-none hover:text-zinc-300"
      >
        ✕
      </button>
      <img
        src={src}
        alt="フルサイズ画像"
        className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// MarkdownContent
// ---------------------------------------------------------------------------

/**
 * Markdownテキストをレンダリングするコンポーネント。
 * remark-gfm で表・打ち消し線・チェックボックスなどのGFM拡張に対応する。
 * remark-breaks で改行をそのまま反映する。
 * コードブロックは react-syntax-highlighter (oneDark) でシンタックスハイライトする。
 */
export function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkBreaks]}
      components={{
        code({ className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          const codeText = String(children).replace(/\n$/, "");
          if (match) {
            return (
              <SyntaxHighlighter
                style={oneDark}
                language={match[1]}
                PreTag="div"
                customStyle={{ borderRadius: "0.5rem", fontSize: "0.8rem", margin: "0.5rem 0" }}
              >
                {codeText}
              </SyntaxHighlighter>
            );
          }
          return (
            <code className="bg-zinc-700 text-zinc-200 rounded px-1 py-0.5 font-mono text-[0.85em]" {...props}>
              {children}
            </code>
          );
        },
        p({ children }) {
          return <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>;
        },
        strong({ children }) {
          return <strong className="font-bold text-zinc-50">{children}</strong>;
        },
        em({ children }) {
          return <em className="italic">{children}</em>;
        },
        h1({ children }) { return <h1 className="text-lg font-bold mt-3 mb-1">{children}</h1>; },
        h2({ children }) { return <h2 className="text-base font-bold mt-3 mb-1">{children}</h2>; },
        h3({ children }) { return <h3 className="text-sm font-bold mt-2 mb-1">{children}</h3>; },
        ul({ children }) { return <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>; },
        ol({ children }) { return <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>; },
        hr() { return <hr className="border-zinc-600 my-3" />; },
        blockquote({ children }) {
          return (
            <blockquote className="border-l-2 border-zinc-500 pl-3 text-zinc-400 my-2">
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
