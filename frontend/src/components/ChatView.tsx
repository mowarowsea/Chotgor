/**
 * チャットビューコンポーネント。
 * メッセージ一覧の表示・メッセージ送信フォームを担当する。
 */
import { useEffect, useRef, useState } from "react";
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
  /** メッセージ送信コールバック */
  onSend: (content: string) => void;
}

/** メッセージを整形してバブル表示するチャットビュー。 */
export default function ChatView({
  messages,
  characterName,
  userName,
  sending,
  streamingContent,
  streamingReasoning,
  onSend,
}: Props) {
  const [input, setInput] = useState("");
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
    onSend(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Shift+Enter で送信、Enter のみは改行
    if (e.key === "Enter" && e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  return (
    <div className="flex flex-col flex-1 h-full overflow-hidden">
      {/* メッセージ一覧 */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && !sending && (
          <p className="text-zinc-500 text-sm text-center mt-16">
            メッセージを送ってみてください
          </p>
        )}

        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            msg={msg}
            characterName={characterName}
            userName={userName}
          />
        ))}

        {/* ストリーミング中: 思考ブロック・想起記憶 + 応答バブル */}
        {sending && (streamingReasoning || streamingContent !== null) && (
          <div className="flex gap-3 items-start">
            <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-xs font-bold shrink-0">
              {characterName.charAt(0)}
            </div>
            <div className="max-w-[70%] space-y-1">
              {/* 思考ブロック（折りたたみ） */}
              {streamingReasoning && (
                <ThinkingBlock content={streamingReasoning} streaming />
              )}
              {/* 応答テキストバブル */}
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
        className="border-t border-zinc-800 px-6 py-4 flex gap-3 items-end"
      >
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
  streaming?: boolean;
}) {
  const [expanded, setExpanded] = useState(true);

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

/** 1件のメッセージをバブル表示するサブコンポーネント。 */
function MessageBubble({
  msg,
  characterName,
  userName,
}: {
  msg: ChatMessage;
  characterName: string;
  userName: string;
}) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div className="flex gap-3 items-start flex-row-reverse">
        <div className="w-8 h-8 rounded-full bg-zinc-600 flex items-center justify-center text-xs font-bold shrink-0">
          {userName.charAt(0)}
        </div>
        <div className="bg-indigo-900 rounded-2xl rounded-tr-sm px-4 py-2.5 text-zinc-100 text-sm max-w-[70%] whitespace-pre-wrap">
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 items-start">
      <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-xs font-bold shrink-0">
        {characterName.charAt(0)}
      </div>
      <div className="bg-zinc-800 rounded-2xl rounded-tl-sm px-4 py-2.5 text-zinc-100 text-sm max-w-[70%] whitespace-pre-wrap">
        {msg.content}
      </div>
    </div>
  );
}
