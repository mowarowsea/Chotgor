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

/** 記憶カテゴリの背景色・文字色マップ（バックエンドの style.css と合わせる）。 */
const CATEGORY_COLORS: Record<string, { bg: string; text: string }> = {
  identity:   { bg: "#2a3a4a", text: "#8ab4d4" },
  user:       { bg: "#3a2a4a", text: "#c48ad4" },
  semantic:   { bg: "#2a4a3a", text: "#8ad4a4" },
  contextual: { bg: "#4a3a2a", text: "#d4a48a" },
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
        className="flex items-start gap-1.5 rounded px-2 py-1 my-0.5"
        style={colors ? { background: colors.bg, color: colors.text } : { color: "#a1a1aa" }}
      >
        <span className="shrink-0 font-bold">[{category}]</span>
        <span>{text}</span>
      </div>
    );
  }
  // 通常行（見出し・思考テキストなど）
  return <div className="text-zinc-500 whitespace-pre-wrap">{line}</div>;
}

/**
 * `[category] content  (score: X.XX)` 形式かどうかを判定する。
 * カテゴリ名はDBに入っている任意の値を許容する。
 */
function isMemoryLine(line: string): boolean {
  return /^\[[\w_]+\] .+\(score: [\d.]+\)$/.test(line);
}

/**
 * 思考ブロック・想起記憶を折りたたみ表示するコンポーネント。
 * 記憶行はカテゴリ別に色付け、LLMスケッチ部分は区切り線で分けて表示する。
 * ストリーミング中は自動展開する。
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

  const lines = content.split("\n").filter((l) => l !== "");
  const memoryLines = lines.filter(isMemoryLine);
  const sketchLines = lines.filter((l) => !isMemoryLine(l));

  return (
    <div className="border border-zinc-700 rounded-xl overflow-hidden text-xs">
      <button
        className="w-full flex items-center gap-1.5 px-3 py-1.5 text-zinc-400 hover:bg-zinc-800/60 transition-colors text-left"
        onClick={() => setExpanded((e) => !e)}
      >
        <span className="text-[10px]">{expanded ? "▼" : "▶"}</span>
        <span>想起した記憶・スケッチ</span>
        {streaming && <span className="animate-pulse ml-1 text-indigo-400">●</span>}
      </button>
      {expanded && (
        <div className="px-3 py-2 font-mono border-t border-zinc-700/60 leading-relaxed">
          {/* 記憶セクション */}
          {memoryLines.length > 0 && (
            <>
              <div className="flex items-center gap-2 mb-2">
                <hr className="flex-1 border-zinc-700" />
                <span className="text-zinc-600 shrink-0">📚 想起した記憶</span>
                <hr className="flex-1 border-zinc-700" />
              </div>
              <div className="mb-2">
                {memoryLines.map((line, i) => (
                  <ReasoningLine key={i} line={line} />
                ))}
              </div>
            </>
          )}
          {/* スケッチセクション */}
          {sketchLines.length > 0 && (
            <>
              <div className="flex items-center gap-2 my-2">
                <hr className="flex-1 border-zinc-700" />
                <span className="text-zinc-600 shrink-0">💭 スケッチ</span>
                <hr className="flex-1 border-zinc-700" />
              </div>
              <div className="text-zinc-500 whitespace-pre-wrap">
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
// CopyButton（内部共通）
// ---------------------------------------------------------------------------

/** テキストをクリップボードにコピーし、完了時に一時的にチェックマークを表示するボタン。 */
function CopyButton({ text, className = "" }: { text: string; className?: string }) {
  const [done, setDone] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      // Secure Context（HTTPS / localhost）では Clipboard API を使用する
      await navigator.clipboard.writeText(text);
    } catch {
      // HTTP環境（スマホからのローカルアクセス等）向けフォールバック
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
// CharacterAvatar
// ---------------------------------------------------------------------------

/**
 * キャラクターアバター。
 * imageUrl が指定されている場合は画像を表示し、未指定またはロード失敗時はイニシャルを表示する。
 */
export function CharacterAvatar({
  characterName,
  imageUrl,
  bgClass = "bg-indigo-600",
}: {
  characterName: string;
  /** アバター画像URL（省略時はイニシャル表示）。 */
  imageUrl?: string;
  /** イニシャル表示時の背景色 Tailwind クラス。 */
  bgClass?: string;
}) {
  const [imgFailed, setImgFailed] = useState(false);
  return (
    <div className={`w-8 h-8 rounded-full ${bgClass} flex items-center justify-center text-xs font-bold shrink-0 overflow-hidden`}>
      {imageUrl && !imgFailed ? (
        <img
          src={imageUrl}
          alt=""
          className="w-full h-full object-cover"
          onError={() => setImgFailed(true)}
        />
      ) : (
        characterName.charAt(0)
      )}
    </div>
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
  imageUrl,
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
  /** アバター画像URL（省略時はイニシャル表示）。 */
  imageUrl?: string;
}) {
  return (
    <div className="group" data-testid="character-bubble">
      {/* スマホ: アバター+名前を上段に、バブルを下段にフルwidth表示
          デスクトップ: アバター左・バブル右の横並び（従来レイアウト） */}
      <div className="flex flex-col sm:flex-row gap-1 sm:gap-3 items-start">
        {/* アバター行: スマホはアバター+名前を横並び、デスクトップはアバターのみ */}
        <div className="flex items-center gap-2">
          <CharacterAvatar characterName={characterName} imageUrl={imageUrl} bgClass={avatarBg} />
          <p className={`text-xs font-medium ${nameColor} sm:hidden`}>{characterName}</p>
        </div>
        {/* バブルコンテナ: スマホはフルwidth、デスクトップは90%まで */}
        <div className="w-full sm:max-w-[90%] min-w-0 space-y-0.5">
          <p className={`hidden sm:block text-xs font-medium ${nameColor} px-1`}>{characterName}</p>
          {reasoning && <ThinkingBlock content={reasoning} />}
          <div className="bg-zinc-800 rounded-2xl rounded-tl-sm px-4 py-2.5 text-zinc-100 text-sm overflow-hidden">
            <MarkdownContent content={content} />
          </div>
        </div>
      </div>
      {/* アクションエリア: スマホはインデントなし、デスクトップはアバター幅分インデント */}
      {!sending && (
        <div className="pl-0 sm:pl-11 flex items-center gap-1">
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
    <div className="group">
      {/* スマホ: アバター+名前を上段に、バブルを下段にフルwidth表示
          デスクトップ: アバター右・バブル左の横並び（従来レイアウト） */}
      <div className="flex flex-col sm:flex-row-reverse sm:gap-3 items-start">
        {/* アバター行: スマホはアバター+名前を右寄り横並び、デスクトップはアバターのみ */}
        <div className="flex flex-row-reverse sm:flex-row items-center gap-2 mb-1 sm:mb-0 w-full sm:w-auto">
          <div className="w-8 h-8 rounded-full bg-zinc-600 flex items-center justify-center text-xs font-bold shrink-0">
            {userName.charAt(0)}
          </div>
          <p className="text-xs font-medium text-zinc-400 sm:hidden">{userName}</p>
        </div>
        {/* バブルコンテナ: スマホはフルwidth、デスクトップは右寄り90%まで */}
        <div className={`w-full sm:max-w-[90%] sm:flex sm:flex-col sm:items-end min-w-0 ${editing ? "" : ""}`}>
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
          <div className="flex items-end gap-1 flex-row-reverse min-w-0 w-full">
            <div className="bg-indigo-900 rounded-2xl rounded-tr-sm px-4 py-2.5 text-zinc-100 text-sm overflow-hidden">
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
// CodeBlock（コードブロック + 右肩コピーボタン）
// ---------------------------------------------------------------------------

/**
 * コードブロック表示コンポーネント。
 * 右上にコピーボタンを重ねて表示する。言語ラベルも表示する。
 */
function CodeBlock({
  language,
  codeText,
  children,
}: {
  /** 言語名（SyntaxHighlighter 用。未指定時は undefined）。 */
  language?: string;
  /** コピー対象のプレーンテキスト。 */
  codeText: string;
  /** レンダリング済みコンテンツ（SyntaxHighlighter or plain code）。 */
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
      {/* 言語ラベル + コピーボタン（右肩） */}
      <div className="absolute top-2 right-2 flex items-center gap-1.5 z-10">
        {language && (
          <span className="text-[10px] text-zinc-500 font-mono select-none">{language}</span>
        )}
        <button
          onClick={handleCopy}
          title="コピー"
          className="opacity-0 group-hover/code:opacity-100 transition-opacity bg-zinc-700 hover:bg-zinc-600 text-zinc-400 hover:text-zinc-200 rounded p-1"
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
          // フェンスコードブロックの children は末尾に \n が付く。インラインコードは付かない。
          const rawCode = String(children);
          const isBlock = rawCode.endsWith("\n");
          const codeText = rawCode.replace(/\n$/, "");

          if (match) {
            // 言語指定ありのコードブロック → シンタックスハイライト
            return (
              <CodeBlock language={match[1]} codeText={codeText}>
                <SyntaxHighlighter
                  style={oneDark}
                  language={match[1]}
                  PreTag="div"
                  customStyle={{ borderRadius: "0.5rem", fontSize: "0.8rem", margin: 0, overflowX: "auto" }}
                >
                  {codeText}
                </SyntaxHighlighter>
              </CodeBlock>
            );
          }

          if (isBlock) {
            // 言語指定なしのコードブロック → シンプルなブロック表示
            return (
              <CodeBlock codeText={codeText}>
                <div className="bg-zinc-900 rounded-lg p-3 overflow-x-auto">
                  <code className="font-mono text-[0.8rem] text-zinc-200 whitespace-pre">{codeText}</code>
                </div>
              </CodeBlock>
            );
          }

          // インラインコード
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
