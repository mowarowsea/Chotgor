/**
 * グループチャットビューコンポーネント。
 * 複数キャラクターの発言を色分けバブルで表示し、ユーザーのメッセージ送信を担当する。
 */
import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../api";
import { CharacterBubble, UserBubble } from "./ChatBubbles";

/** キャラクターごとのカラーパレット（アバター背景色・名前ラベル色）。 */
const CHAR_COLORS = [
  { bg: "bg-indigo-600",  text: "text-indigo-400"  },
  { bg: "bg-emerald-600", text: "text-emerald-400" },
  { bg: "bg-violet-600",  text: "text-violet-400"  },
  { bg: "bg-amber-600",   text: "text-amber-400"   },
  { bg: "bg-rose-600",    text: "text-rose-400"    },
  { bg: "bg-cyan-600",    text: "text-cyan-400"    },
];

interface Props {
  /** 表示するメッセージ一覧 */
  messages: ChatMessage[];
  /** グループ参加者のキャラクター名リスト（色割り当て順序に使用） */
  participantNames: string[];
  /** ユーザ名（表示用） */
  userName: string;
  /** 送信処理中フラグ */
  sending: boolean;
  /** 応答待機中のキャラクター名（null = 待機なし） */
  waitingCharacter: string | null;
  /** メッセージIDに紐付いた thinking/reasoning テキスト */
  reasoningMap: Record<string, string>;
  /** メッセージ送信コールバック */
  onSend: (content: string) => void;
  /**
   * ユーザメッセージ編集・キャラクター応答再生成コールバック。
   * fromMessageId 以降を削除して content で再送する。
   */
  onRetry?: (fromMessageId: string, content: string) => void;
}

/** グループチャットのメッセージ一覧と送信フォームを表示するビュー。 */
export default function GroupChatView({
  messages,
  participantNames,
  userName,
  sending,
  waitingCharacter,
  reasoningMap,
  onSend,
  onRetry,
}: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  /** キャラクター名からカラーパレットのインデックスを返す。 */
  const getCharColor = (charName: string) => {
    const idx = participantNames.indexOf(charName);
    return CHAR_COLORS[idx >= 0 ? idx % CHAR_COLORS.length : 0];
  };

  /** メッセージ追加・送信待機時は最下部へスクロールする。 */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, waitingCharacter]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    onSend(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  return (
    <div className="flex flex-col flex-1 h-full overflow-hidden">
      {/* メッセージ一覧 */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-6 py-4 space-y-3">
        {messages.length === 0 && !sending && (
          <p className="text-zinc-500 text-sm text-center mt-16">
            グループチャットを始めましょう
          </p>
        )}

        {messages.map((msg, idx) => {
          if (msg.role === "user") {
            return (
              <UserBubble
                key={msg.id}
                content={msg.content}
                userName={userName}
                sending={sending}
                onEdit={onRetry ? (newContent) => onRetry(msg.id, newContent) : undefined}
              />
            );
          }

          // キャラクターメッセージ: 共通 CharacterBubble を使用
          const charName = msg.character_name ?? participantNames[0] ?? "キャラクター";
          const color = getCharColor(charName);

          return (
            <CharacterBubble
              key={msg.id}
              characterName={charName}
              content={msg.content}
              reasoning={reasoningMap[msg.id]}
              avatarBg={color.bg}
              nameColor={color.text}
              sending={sending}
              onRegenerate={onRetry ? () => {
                // 直前のユーザメッセージを逆順で探す
                const precedingUser = [...messages]
                  .slice(0, idx)
                  .reverse()
                  .find((m) => m.role === "user");
                if (precedingUser) {
                  onRetry(precedingUser.id, precedingUser.content);
                }
              } : undefined}
            />
          );
        })}

        {/* 応答待機中スピナー */}
        {waitingCharacter && (() => {
          const color = getCharColor(waitingCharacter);
          return (
            <div className="flex gap-3 items-start">
              <div className={`w-8 h-8 rounded-full ${color.bg} flex items-center justify-center text-xs font-bold shrink-0`}>
                {waitingCharacter.charAt(0)}
              </div>
              <div className="space-y-0.5">
                <p className={`text-xs font-medium ${color.text} px-1`}>{waitingCharacter}</p>
                <div className="bg-zinc-800 rounded-2xl rounded-tl-sm px-4 py-2.5 text-zinc-400 text-sm">
                  <span className="animate-pulse">考え中...</span>
                </div>
              </div>
            </div>
          );
        })()}

        <div ref={bottomRef} />
      </div>

      {/* 入力フォーム */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-zinc-800 px-3 sm:px-6 py-3 sm:py-4 flex gap-3 items-end"
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
