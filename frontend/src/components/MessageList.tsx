/**
 * メッセージ一覧表示共通コンポーネント。
 * メッセージのループ表示、自動スクロール、キャラクターごとのカラー設定、考え中インジケーターを管理する。
 */
import { useEffect, useRef } from "react";
import type { ChatMessage } from "../api";
import { CharacterBubble, CharacterAvatar, UserBubble, ThinkingBlock, Bubble } from "./ChatBubbles";

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
    /** char_msg_id → log_message_id のマッピング。バブルのログ折りたたみに使用する。 */
    msgLogIds?: Record<string, string>;
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
    msgLogIds = {},
}: Props) {
    const bottomRef = useRef<HTMLDivElement>(null);

    /**
     * キャラクター別配色バブル（cb0〜cb9）を使うかどうか。
     * グループチャット（参加者名リストあり）でのみ true。1on1 はニュートラル面。
     */
    const colored = participantNames.length > 0;

    /** メッセージ追加・ストリーミング・待機中は最下部へスクロールする。 */
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, sending, streamingContent, waitingCharacter]);

    return (
        <div className="flex-1 overflow-y-auto overflow-x-hidden">
          {/* pt-16: 浮遊ヘッダー分の上余白。 */}
          <div className="max-w-[760px] mx-auto px-4 sm:px-6 pt-16 pb-6 space-y-5">
            {messages.length === 0 && !sending && !waitingCharacter && (
                <p className="text-ch-t4 text-xs text-center mt-20">
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

                return (
                    <CharacterBubble
                        key={msg.id}
                        characterName={charName}
                        presetName={msg.preset_name}
                        content={msg.content}
                        reasoning={reasoningMap[msg.id]}
                        colored={colored}
                        sending={sending}
                        logMessageId={msgLogIds[msg.id]}
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

            {/* ストリーミング中 */}
            {sending && (streamingReasoning || (streamingContent !== null && streamingContent.trim().length > 0)) && (() => {
                const streamCharName = waitingCharacter ?? characterName;
                return (
                    <div className="flex gap-2.5 max-w-[88%]">
                        <CharacterAvatar characterName={streamCharName} size={28} />
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1.5 mb-1 text-[11px]">
                                <span className="font-semibold text-ch-t2">{streamCharName}</span>
                            </div>
                            {streamingReasoning && (
                                <div className="mb-1"><ThinkingBlock content={streamingReasoning} streaming /></div>
                            )}
                            {streamingContent !== null && streamingContent.trim().length > 0 && (
                                <Bubble kind="character" colored={colored} characterName={streamCharName}>
                                    <span className="whitespace-pre-wrap">{streamingContent}</span>
                                    <span className="animate-pulse inline-block ml-0.5 text-ch-accent">▌</span>
                                </Bubble>
                            )}
                        </div>
                    </div>
                );
            })()}

            {/* 応答待機インジケーター */}
            {(waitingCharacter || sending) && !streamingReasoning && (streamingContent === null || streamingContent.trim().length === 0) && (() => {
                const charName = waitingCharacter ?? characterName;
                return (
                    <div className="flex gap-2.5">
                        <CharacterAvatar characterName={charName} size={28} />
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1.5 mb-1 text-[11px]">
                                <span className="font-semibold text-ch-t2">{charName}</span>
                            </div>
                            <span className="text-ch-t3 text-sm animate-pulse">…</span>
                        </div>
                    </div>
                );
            })()}

            <div ref={bottomRef} />
          </div>
        </div>
    );
}
