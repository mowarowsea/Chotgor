/**
 * チャットビューコンポーネント。
 * メッセージ一覧の表示・メッセージ送信フォームを担当する。
 */
import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../api";
import { CharacterBubble, UserBubble, ThinkingBlock } from "./ChatBubbles";

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
          msg.role === "user" ? (
            <UserBubble
              key={msg.id}
              content={msg.content}
              userName={userName}
              images={msg.images}
              sending={sending}
              onEdit={(newContent) => onRetry(msg.id, newContent, [])}
            />
          ) : (
            <CharacterBubble
              key={msg.id}
              characterName={characterName}
              content={msg.content}
              reasoning={reasoningMap[msg.id]}
              sending={sending}
              onRegenerate={() => {
                // 直前のユーザメッセージを逆順で探して画像IDも引き継ぐ
                const precedingUser = [...messages]
                  .slice(0, idx)
                  .reverse()
                  .find((m) => m.role === "user");
                if (precedingUser) {
                  onRetry(precedingUser.id, precedingUser.content, precedingUser.images ?? []);
                }
              }}
            />
          )
        ))}

        {/* ストリーミング中: 思考ブロック・想起記憶 + 応答バブル */}
        {sending && (streamingReasoning || streamingContent !== null) && (
          <div className="flex gap-3 items-start">
            <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-xs font-bold shrink-0">
              {characterName.charAt(0)}
            </div>
            <div className="max-w-[85%] sm:max-w-[70%] space-y-1">
              <p className="text-xs font-medium text-indigo-400 px-1">{characterName}</p>
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

