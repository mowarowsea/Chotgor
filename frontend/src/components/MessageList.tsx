/**
 * メッセージ一覧表示共通コンポーネント。
 * メッセージのループ表示、自動スクロール、キャラクターごとのカラー設定、考え中インジケーターを管理する。
 */
import { useEffect, useRef } from "react";
import type { ChatMessage } from "../api";
import { CharacterBubble, CharacterAvatar, UserBubble, ThinkingBlock } from "./ChatBubbles";

/**
 * キャラクターごとのカラーパレット。
 * バックエンドのバッジカラーに合わせた霧中の翡翠系。
 */
/**
 * キャラクターごとのカラーパレット。
 * キャラクター名にのみ使用する色。背景はすべてニュートラルサーフェス (bg-ch-s2) を共用する。
 */
const CHAR_COLORS = [
  { bg: "bg-ch-s2", text: "text-ch-accent-t" },         // jade green (アクセント)
  { bg: "bg-ch-s2", text: "text-[#6090c0]" },           // steel blue
  { bg: "bg-ch-s2", text: "text-[#a878c8]" },           // muted violet
  { bg: "bg-ch-s2", text: "text-[#c89060]" },           // amber
  { bg: "bg-ch-s2", text: "text-[#c87090]" },           // rose
  { bg: "bg-ch-s2", text: "text-[#60a8c0]" },           // cyan
];

interface Props {
    /** 表示するメッセージ一覧 */
    messages: ChatMessage[];
    /** ユーザ名（表示用） */
    userName: string;
    /** スクロール方向変化コールバック。下スクロールで false、上スクロールで true を渡す。 */
    onHeaderVisibilityChange?: (visible: boolean) => void;
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
    /** キャラクター名→IDのマップ。アバター画像URLの生成に使用する。 */
    characterIdMap?: Record<string, string>;
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
    onHeaderVisibilityChange,
    onRetry,
    characterIdMap = {},
}: Props) {
    const bottomRef = useRef<HTMLDivElement>(null);
    /** スクロール方向検知用: 直前のスクロール位置を記録する。 */
    const lastScrollYRef = useRef(0);
    /** 自動スクロール中フラグ。プログラム起因のスクロールイベントでヘッダーが暴れないよう抑制する。 */
    const autoScrollingRef = useRef(false);
    /** 自動スクロール終了タイマーID。 */
    const autoScrollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    /**
     * ヘッダーアニメーション中フラグ。
     * ヘッダー表示切り替え後 400ms はスクロール判定をロックし、
     * アニメーションによるレイアウトリフロー → スクロールイベント → ヘッダー再切り替えの
     * フィードバックループを断ち切る。
     */
    const headerTransitioningRef = useRef(false);
    /** ヘッダーアニメーションロック解除タイマーID。 */
    const headerTransitionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    /** キャラクター名からアバター画像URLを生成する。IDが不明な場合は undefined を返す。 */
    const getCharImageUrl = (charName: string): string | undefined => {
        const id = characterIdMap[charName];
        return id ? `/api/characters/${id}/image` : undefined;
    };

    /** キャラクター名からカラーパレットのインデックスを返す。 */
    const getCharColor = (charName: string) => {
        if (participantNames.length === 0) {
            return CHAR_COLORS[0];
        }
        const idx = participantNames.indexOf(charName);
        return CHAR_COLORS[idx >= 0 ? idx % CHAR_COLORS.length : 0];
    };

    /** メッセージ追加・ストリーミング・待機中は最下部へスクロールする。 */
    useEffect(() => {
        autoScrollingRef.current = true;
        if (autoScrollTimerRef.current) clearTimeout(autoScrollTimerRef.current);
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        autoScrollTimerRef.current = setTimeout(() => {
            autoScrollingRef.current = false;
        }, 600);
    }, [messages, sending, streamingContent, waitingCharacter]);

    /**
     * ヘッダー表示切り替えをトリガーし、アニメーション完了まで再トリガーをロックする。
     */
    const triggerHeaderChange = (visible: boolean) => {
        headerTransitioningRef.current = true;
        if (headerTransitionTimerRef.current) clearTimeout(headerTransitionTimerRef.current);
        headerTransitionTimerRef.current = setTimeout(() => {
            headerTransitioningRef.current = false;
        }, 400);
        onHeaderVisibilityChange!(visible);
    };

    /** スクロール方向を検知してヘッダー表示状態をコールバックに通知する。 */
    const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
        if (!onHeaderVisibilityChange) return;
        if (autoScrollingRef.current) return;
        if (headerTransitioningRef.current) return;
        const currentY = e.currentTarget.scrollTop;
        if (currentY < 30) {
            triggerHeaderChange(true);
            lastScrollYRef.current = currentY;
            return;
        }
        if (currentY < lastScrollYRef.current) {
            triggerHeaderChange(true);
        } else if (currentY > lastScrollYRef.current + 30) {
            triggerHeaderChange(false);
        } else {
            return;
        }
        lastScrollYRef.current = currentY;
    };

    return (
        <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 sm:px-8 py-6 space-y-6" onScroll={handleScroll}>
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
                const displayName = msg.preset_name ? `${charName}@${msg.preset_name}` : charName;
                const color = getCharColor(charName);

                return (
                    <CharacterBubble
                        key={msg.id}
                        characterName={displayName}
                        content={msg.content}
                        reasoning={reasoningMap[msg.id]}
                        avatarBg={color.bg}
                        nameColor={color.text}
                        sending={sending}
                        imageUrl={getCharImageUrl(charName)}
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
                const color = getCharColor(streamCharName);
                return (
                    <div>
                        {/* ヘッダー行 */}
                        <div className="flex items-center gap-3 mb-2">
                            <CharacterAvatar characterName={streamCharName} imageUrl={getCharImageUrl(streamCharName)} bgClass={color.bg} />
                            <span className={`text-xs font-medium ${color.text}`}>{streamCharName}</span>
                        </div>
                        {/* コンテンツ */}
                        <div className="pl-[72px] space-y-1">
                            {streamingReasoning && (
                                <ThinkingBlock content={streamingReasoning} streaming />
                            )}
                            {streamingContent !== null && streamingContent.trim().length > 0 && (
                                <div className="text-ch-t1 text-sm leading-relaxed whitespace-pre-wrap">
                                    {streamingContent}
                                    <span className="animate-pulse inline-block ml-0.5 text-ch-accent-t">▌</span>
                                </div>
                            )}
                        </div>
                    </div>
                );
            })()}

            {/* 応答待機インジケーター */}
            {(waitingCharacter || sending) && !streamingReasoning && (streamingContent === null || streamingContent.trim().length === 0) && (() => {
                const charName = waitingCharacter ?? characterName;
                const color = getCharColor(charName);
                return (
                    <div>
                        <div className="flex items-center gap-2 mb-1.5">
                            <CharacterAvatar characterName={charName} imageUrl={getCharImageUrl(charName)} bgClass={color.bg} />
                            <span className={`text-xs font-medium ${color.text}`}>{charName}</span>
                        </div>
                        <div className="pl-[72px]">
                            <span className="text-ch-t3 text-sm animate-pulse">…</span>
                        </div>
                    </div>
                );
            })()}

            <div ref={bottomRef} />
        </div>
    );
}
