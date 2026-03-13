/**
 * メッセージ一覧表示共通コンポーネント。
 * メッセージのループ表示、自動スクロール、キャラクターごとのカラー設定、考え中インジケーターを管理する。
 */
import { useEffect, useRef } from "react";
import type { ChatMessage } from "../api";
import { CharacterBubble, UserBubble, ThinkingBlock } from "./ChatBubbles";

/** キャラクターごとのカラーパレット。GroupChatView から移設。 */
const CHAR_COLORS = [
    { bg: "bg-indigo-600", text: "text-indigo-400" },
    { bg: "bg-emerald-600", text: "text-emerald-400" },
    { bg: "bg-violet-600", text: "text-violet-400" },
    { bg: "bg-amber-600", text: "text-amber-400" },
    { bg: "bg-rose-600", text: "text-rose-400" },
    { bg: "bg-cyan-600", text: "text-cyan-400" },
];

interface Props {
    /** 表示するメッセージ一覧 */
    messages: ChatMessage[];
    /** ユーザ名（表示用） */
    userName: string;
    /** 送信処理中フラグ */
    sending: boolean;
    /** 完了済みメッセージIDと reasoning テキストの対応マップ */
    reasoningMap: Record<string, string>;
    /** グループ参加者のキャラクター名リスト（色割り当て用） */
    participantNames?: string[];
    /** ストリーミング中のキャラクター応答テキスト */
    streamingContent?: string | null;
    /** ストリーミング中の思考ブロック・想起記憶テキスト */
    streamingReasoning?: string | null;
    /** 応答待機中のキャラクター名（グループチャット用） */
    waitingCharacter?: string | null;
    /** 1on1チャットのデフォルトキャラクター名 */
    characterName?: string;
    /** 空の状態の時のメッセージ */
    emptyMessage?: string;
    /** メッセージ編集・再生成時のコールバック */
    onRetry?: (fromMessageId: string, content: string, imageIds: string[]) => void;
}

/**
 * チャットメッセージのリストを表示し、自動スクロールを制御するコンポーネント。
 */
export default function MessageList({
    messages,
    userName,
    sending,
    reasoningMap,
    participantNames = [],
    streamingContent = null,
    streamingReasoning = null,
    waitingCharacter = null,
    characterName = "キャラクター",
    emptyMessage = "メッセージを送ってみてください",
    onRetry,
}: Props) {
    const bottomRef = useRef<HTMLDivElement>(null);

    /** キャラクター名からカラーパレットのインデックスを返す。 */
    const getCharColor = (charName: string) => {
        if (participantNames.length === 0) {
            return { bg: "bg-indigo-600", text: "text-indigo-400" };
        }
        const idx = participantNames.indexOf(charName);
        return CHAR_COLORS[idx >= 0 ? idx % CHAR_COLORS.length : 0];
    };

    /** メッセージ追加・ストリーミング・待機中は最下部へスクロールする。 */
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, sending, streamingContent, waitingCharacter]);

    return (
        <div className="flex-1 overflow-y-auto px-3 sm:px-6 py-4 space-y-4">
            {messages.length === 0 && !sending && !waitingCharacter && (
                <p className="text-zinc-500 text-sm text-center mt-16">
                    {emptyMessage}
                </p>
            )}

            {messages.map((msg, idx) => {
                if (msg.role === "user") {
                    return (
                        <UserBubble
                            key={msg.id}
                            content={msg.content}
                            userName={userName}
                            images={msg.images}
                            sending={sending}
                            onEdit={onRetry ? (newContent) => onRetry(msg.id, newContent, msg.images ?? []) : undefined}
                        />
                    );
                }

                const charName = msg.character_name ?? characterName;
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
                            const precedingUser = [...messages]
                                .slice(0, idx)
                                .reverse()
                                .find((m) => m.role === "user");
                            if (precedingUser) {
                                onRetry(precedingUser.id, precedingUser.content, precedingUser.images ?? []);
                            }
                        } : undefined}
                    />
                );
            })}

            {/* ストリーミング中: 1on1 は characterName、グループは waitingCharacter の名前・色を使う */}
            {sending && (streamingReasoning || (streamingContent !== null && streamingContent.trim().length > 0)) && (() => {
                const streamCharName = waitingCharacter ?? characterName;
                const color = getCharColor(streamCharName);
                return (
                    <div className="flex gap-3 items-start">
                        <div className={`w-8 h-8 rounded-full ${color.bg} flex items-center justify-center text-xs font-bold shrink-0`}>
                            {streamCharName.charAt(0)}
                        </div>
                        <div className="max-w-[85%] sm:max-w-[70%] space-y-1">
                            <p className={`text-xs font-medium ${color.text} px-1`}>{streamCharName}</p>
                            {streamingReasoning && (
                                <ThinkingBlock content={streamingReasoning} streaming />
                            )}
                            {streamingContent !== null && streamingContent.trim().length > 0 && (
                                <div className="bg-zinc-800 rounded-2xl rounded-tl-sm px-4 py-2.5 text-zinc-100 text-sm whitespace-pre-wrap">
                                    {streamingContent}
                                    <span className="animate-pulse inline-block ml-0.5 text-indigo-400">▌</span>
                                </div>
                            )}
                        </div>
                    </div>
                );
            })()}

            {/* 応答待機スピナー: ストリーミング内容がまだない場合のみ表示（ストリーミング中は非表示） */}
            {(waitingCharacter || sending) && !streamingReasoning && (streamingContent === null || streamingContent.trim().length === 0) && (() => {
                const charName = waitingCharacter ?? characterName;
                const color = getCharColor(charName);
                return (
                    <div className="flex gap-3 items-start">
                        <div className={`w-8 h-8 rounded-full ${color.bg} flex items-center justify-center text-xs font-bold shrink-0`}>
                            {charName.charAt(0)}
                        </div>
                        <div className="space-y-0.5">
                            <p className={`text-xs font-medium ${color.text} px-1`}>{charName}</p>
                            <div className="bg-zinc-800 rounded-2xl rounded-tl-sm px-4 py-2.5 text-zinc-400 text-sm">
                                <span className="animate-pulse">考え中...</span>
                            </div>
                        </div>
                    </div>
                );
            })()}

            <div ref={bottomRef} />
        </div>
    );
}
