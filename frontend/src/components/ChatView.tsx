/**
 * チャットビューコンポーネント。
 * メッセージ一覧の表示・メッセージ送信フォームを担当する。
 */
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { ChatMessage } from "../api";

interface Props {
  /** 表示するメッセージ一覧 */
  messages: ChatMessage[];
  /** キャラクター名（表示用） */
  characterName: string;
  /** ユーザ名（表示用） */
  userName: string;
  /** 送信処理中フラグ */
  sending: boolean;
  /** ストリーミング中のキャラクター応答テキスト（null = ストリーミングなし） */
  streamingContent: string | null;
  /** ストリーミング中の思考ブロック・想起記憶テキスト（null = なし） */
  streamingReasoning: string | null;
  /** 完了済みメッセージIDと reasoning テキストの対応マップ */
  reasoningMap: Record<string, string>;
  /**
   * メッセージ送信コールバック。
   * files には添付画像ファイルを渡す（空配列可）。
   */
  onSend: (content: string, files: File[]) => void;
  /**
   * ユーザメッセージ編集・キャラクター応答再生成コールバック。
   * fromMessageId 以降を削除して content で再送する。
   * imageIds には再送する画像IDリストを渡す（再生成時は元メッセージの画像を引き継ぐ）。
   */
  onRetry: (fromMessageId: string, content: string, imageIds: string[]) => void;
}

/** メッセージを整形してバブル表示するチャットビュー。 */
export default function ChatView({
  messages,
  characterName,
  userName,
  sending,
  streamingContent,
  streamingReasoning,
  reasoningMap,
  onSend,
  onRetry,
}: Props) {
  const [input, setInput] = useState("");
  /** 送信前の添付ファイルリスト */
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  /** メッセージ追加・ストリーミング中は最下部へスクロールする。 */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending, streamingContent]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    const files = pendingFiles;
    setPendingFiles([]);
    onSend(text, files);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Shift+Enter で送信、Enter のみは改行
    if (e.key === "Enter" && e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files ?? []);
    if (selected.length === 0) return;
    setPendingFiles((prev) => [...prev, ...selected]);
    // 同じファイルを再選択できるように value をリセットする
    e.target.value = "";
  };

  const removePendingFile = (idx: number) => {
    setPendingFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  return (
    <div className="flex flex-col flex-1 h-full overflow-hidden">
      {/* メッセージ一覧 */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-6 py-4 space-y-4">
        {messages.length === 0 && !sending && (
          <p className="text-zinc-500 text-sm text-center mt-16">
            メッセージを送ってみてください
          </p>
        )}

        {messages.map((msg, idx) => (
          <MessageBubble
            key={msg.id}
            msg={msg}
            characterName={characterName}
            userName={userName}
            reasoning={reasoningMap[msg.id]}
            sending={sending}
            onEdit={
              msg.role === "user"
                ? (newContent) => onRetry(msg.id, newContent, [])
                : undefined
            }
            onRegenerate={
              msg.role === "character"
                ? () => {
                    // 直前のユーザメッセージを逆順で探して画像IDも引き継ぐ
                    const precedingUser = [...messages]
                      .slice(0, idx)
                      .reverse()
                      .find((m) => m.role === "user");
                    if (precedingUser) {
                      onRetry(precedingUser.id, precedingUser.content, precedingUser.images ?? []);
                    }
                  }
                : undefined
            }
          />
        ))}

        {/* ストリーミング中: 思考ブロック・想起記憶 + 応答バブル */}
        {sending && (streamingReasoning || streamingContent !== null) && (
          <div className="flex gap-3 items-start">
            <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-xs font-bold shrink-0">
              {characterName.charAt(0)}
            </div>
            <div className="max-w-[85%] sm:max-w-[85%] sm:max-w-[70%] space-y-1">
              {/* 思考ブロック（折りたたみ） */}
              {streamingReasoning && (
                <ThinkingBlock content={streamingReasoning} streaming />
              )}
              {/* 応答テキストバブル（ストリーミング中はプレーンテキスト表示） */}
              {streamingContent !== null && streamingContent.trim().length > 0 && (
                <div className="bg-zinc-800 rounded-2xl rounded-tl-sm px-4 py-2.5 text-zinc-100 text-sm whitespace-pre-wrap">
                  {streamingContent}
                  <span className="animate-pulse inline-block ml-0.5 text-indigo-400">▌</span>
                </div>
              )}
              {/* 応答待機スピナー（reasoning もテキストもまだない場合） */}
              {!streamingReasoning && (streamingContent === null || streamingContent.trim().length === 0) && (
                <div className="bg-zinc-800 rounded-2xl rounded-tl-sm px-4 py-2.5 text-zinc-400 text-sm">
                  <span className="animate-pulse">考え中...</span>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* 入力フォーム */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-zinc-800 px-3 sm:px-6 py-3 sm:py-4 flex flex-col gap-2"
      >
        {/* 添付画像サムネイルプレビュー */}
        {pendingFiles.length > 0 && (
          <div className="flex gap-2 flex-wrap">
            {pendingFiles.map((file, idx) => (
              <div key={idx} className="relative group/thumb">
                <img
                  src={URL.createObjectURL(file)}
                  alt={file.name}
                  className="w-16 h-16 object-cover rounded-lg border border-zinc-700"
                />
                <button
                  type="button"
                  onClick={() => removePendingFile(idx)}
                  className="absolute -top-1 -right-1 w-4 h-4 bg-zinc-600 hover:bg-zinc-500 rounded-full text-[10px] text-white flex items-center justify-center leading-none"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-3 items-end">
          {/* 画像添付ボタン */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={handleFileChange}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={sending}
            title="画像を添付"
            className="text-zinc-500 hover:text-zinc-300 disabled:opacity-40 transition-colors p-2 rounded-lg hover:bg-zinc-800 shrink-0 self-end mb-0.5"
          >
            {/* Heroicons: paper-clip */}
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={20} height={20}>
              <path strokeLinecap="round" strokeLinejoin="round" d="m18.375 12.739-7.693 7.693a4.5 4.5 0 0 1-6.364-6.364l10.94-10.94A3 3 0 1 1 19.5 7.372L8.552 18.32m.009-.01-.01.01m5.699-9.941-7.81 7.81a1.5 1.5 0 0 0 2.112 2.13" />
            </svg>
          </button>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="メッセージを入力… (Shift+Enter で送信)"
            rows={3}
            disabled={sending}
            className="flex-1 bg-zinc-800 text-zinc-100 placeholder-zinc-500 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || sending}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-xl px-5 py-3 text-sm font-medium transition-colors h-fit"
          >
            送信
          </button>
        </div>
      </form>
    </div>
  );
}

/**
 * 思考ブロック・想起記憶を折りたたみ表示するサブコンポーネント。
 * ストリーミング中は自動展開し、完了後は折りたたみ可能にする。
 */
function ThinkingBlock({
  content,
  streaming = false,
}: {
  content: string;
  /** true のときはストリーミング中（自動展開・アニメーション表示）。false のときは折りたたみ初期状態。 */
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

/**
 * 1件のメッセージをバブル表示するサブコンポーネント。
 * ユーザメッセージはホバー時に編集ボタン、キャラクターメッセージは再生成ボタンを表示する。
 */
function MessageBubble({
  msg,
  characterName,
  userName,
  reasoning,
  sending,
  onEdit,
  onRegenerate,
}: {
  msg: ChatMessage;
  characterName: string;
  userName: string;
  /** キャラクターメッセージに紐付いた reasoning テキスト（思考ブロック・想起記憶） */
  reasoning?: string;
  /** 送信処理中フラグ（処理中はボタンを非表示にする） */
  sending: boolean;
  /** ユーザメッセージ編集コールバック（ユーザメッセージのみ） */
  onEdit?: (newContent: string) => void;
  /** キャラクター応答再生成コールバック（キャラクターメッセージのみ） */
  onRegenerate?: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(msg.content);
  const isUser = msg.role === "user";

  /** 編集を確定して送信する。 */
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
      setEditText(msg.content);
    }
  };

  if (isUser) {
    return (
      <div className="flex gap-3 items-start flex-row-reverse group">
        <div className="w-8 h-8 rounded-full bg-zinc-600 flex items-center justify-center text-xs font-bold shrink-0">
          {userName.charAt(0)}
        </div>
        <div className="max-w-[85%] sm:max-w-[70%] flex flex-col items-end gap-1">
          {/* 添付画像グリッド */}
          {msg.images && msg.images.length > 0 && (
            <ImageGrid imageIds={msg.images} />
          )}
          {editing ? (
            /* インライン編集フォーム */
            <div className="flex flex-col gap-2">
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
                  onClick={() => { setEditing(false); setEditText(msg.content); }}
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
            /* 通常表示 + ホバー時に編集ボタン */
            <div className="flex items-end gap-2 flex-row-reverse">
              <div className="bg-indigo-900 rounded-2xl rounded-tr-sm px-4 py-2.5 text-zinc-100 text-sm">
                <MarkdownContent content={msg.content} />
              </div>
              {!sending && onEdit && (
                <button
                  onClick={() => setEditing(true)}
                  title="編集"
                  className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-zinc-500 hover:text-zinc-300 transition-all p-1 rounded shrink-0"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={16} height={16}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10" />
                  </svg>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 items-start group" data-testid="character-bubble">
      <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-xs font-bold shrink-0">
        {characterName.charAt(0)}
      </div>
      <div className="max-w-[85%] sm:max-w-[70%] space-y-1">
        {/* 思考ブロック・想起記憶（完了後は折りたたみ状態で表示） */}
        {reasoning && <ThinkingBlock content={reasoning} />}
        <div className="bg-zinc-800 rounded-2xl rounded-tl-sm px-4 py-2.5 text-zinc-100 text-sm">
          <MarkdownContent content={msg.content} />
        </div>
        {/* 再生成ボタン（モバイルは常時表示、デスクトップはホバー時） */}
        {!sending && onRegenerate && (
          <button
            onClick={onRegenerate}
            title="再生成"
            className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-zinc-500 hover:text-zinc-300 text-xs transition-all px-2 py-1 rounded hover:bg-zinc-800"
          >
            ↺ 再生成
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * 添付画像IDリストをサムネイルグリッドで表示するサブコンポーネント。
 * クリックするとフルサイズモーダルを開く。
 */
function ImageGrid({ imageIds }: { imageIds: string[] }) {
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
function ImageModal({ src, onClose }: { src: string; onClose: () => void }) {
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

/**
 * Markdownテキストをレンダリングするサブコンポーネント。
 * remark-gfm で表・打ち消し線・チェックボックスなどのGFM拡張に対応する。
 * コードブロックは react-syntax-highlighter (oneDark) でシンタックスハイライトする。
 * インラインコードは等幅フォントで薄いグレー背景で表示する。
 * 日本語文中の **太字** など、前後がアルファベット以外の場合も正しく太字になる。
 */
function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkBreaks]}
      components={{
        // コードブロック: 言語付きはシンタックスハイライト、なしはシンプル表示
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
        // 段落: デフォルトの <p> マージンを調整
        p({ children }) {
          return <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>;
        },
        // 太字
        strong({ children }) {
          return <strong className="font-bold text-zinc-50">{children}</strong>;
        },
        // 斜体
        em({ children }) {
          return <em className="italic">{children}</em>;
        },
        // 見出し
        h1({ children }) { return <h1 className="text-lg font-bold mt-3 mb-1">{children}</h1>; },
        h2({ children }) { return <h2 className="text-base font-bold mt-3 mb-1">{children}</h2>; },
        h3({ children }) { return <h3 className="text-sm font-bold mt-2 mb-1">{children}</h3>; },
        // 箇条書き・番号リスト
        ul({ children }) { return <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>; },
        ol({ children }) { return <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>; },
        // 水平線
        hr() { return <hr className="border-zinc-600 my-3" />; },
        // 引用
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
