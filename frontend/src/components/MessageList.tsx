/**
 * メッセージ一覧表示共通コンポーネント。
 * メッセージのループ表示、自動スクロール、キャラクターごとのカラー設定、考え中インジケーターを管理する。
 */
import { useEffect, useRef } from "react";
import type { ChatMessage } from "../api";
import { CharacterBubble, CharacterAvatar, UserBubble, ThinkingBlock } from "./ChatBubbles";

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
            return { bg: "bg-indigo-600", text: "text-indigo-400" };
        }
        const idx = participantNames.indexOf(charName);
        return CHAR_COLORS[idx >= 0 ? idx % CHAR_COLORS.length : 0];
    };

    /** メッセージ追加・ストリーミング・待機中は最下部へスクロールする。
     * スクロール中はヘッダー表示判定を抑制し、プログラム起因のスクロールイベントで暴れないようにする。
     */
    useEffect(() => {
        autoScrollingRef.current = true;
        if (autoScrollTimerRef.current) clearTimeout(autoScrollTimerRef.current);
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        // smooth scroll のアニメーション完了後にフラグを解除する（~500ms）
        autoScrollTimerRef.current = setTimeout(() => {
            autoScrollingRef.current = false;
        }, 600);
    }, [messages, sending, streamingContent, waitingCharacter]);

    /**
     * ヘッダー表示切り替えをトリガーし、アニメーション完了まで再トリガーをロックする。
     * ヘッダーの max-h アニメーション（300ms）がレイアウトリフローを起こしてスクロールイベントを
     * 再発火させるため、400ms のロック期間でフィードバックループを断ち切る。
     */
    const triggerHeaderChange = (visible: boolean) => {
        headerTransitioningRef.current = true;
        if (headerTransitionTimerRef.current) clearTimeout(headerTransitionTimerRef.current);
        // ヘッダー CSS transition は duration-300。その完了を待ってロック解除する。
        headerTransitionTimerRef.current = setTimeout(() => {
            headerTransitioningRef.current = false;
        }, 400);
        onHeaderVisibilityChange!(visible);
    };

    /** スクロール方向を検知してヘッダー表示状態をコールバックに通知する。
     * 自動スクロール中・ヘッダーアニメーション中はスキップしてフィードバックループを防ぐ。
     */
    const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
        if (!onHeaderVisibilityChange) return;
        // 自動スクロール中はヘッダー判定をスキップして振動を防ぐ
        if (autoScrollingRef.current) return;
        // ヘッダーアニメーション中はレイアウトリフロー起因のイベントをすべて無視する
        if (headerTransitioningRef.current) return;
        const currentY = e.currentTarget.scrollTop;
        // スクロール最上部付近は常にヘッダーを表示する
        if (currentY < 30) {
            triggerHeaderChange(true);
            lastScrollYRef.current = currentY;
            return;
        }
        if (currentY < lastScrollYRef.current) {
            // 上スクロール: ヘッダーを表示する
            triggerHeaderChange(true);
        } else if (currentY > lastScrollYRef.current + 30) {
            // 下スクロール: 30px のデッドバンドで誤検知を防ぐ
            triggerHeaderChange(false);
        } else {
            // デッドバンド内: 基準値を更新せず判定もしない
            return;
        }
        lastScrollYRef.current = currentY;
    };

    return (
        <div className="flex-1 overflow-y-auto px-3 sm:px-6 py-4 space-y-4" onScroll={handleScroll}>
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
                // preset_name が存在する場合は "キャラ@プリセット" 形式で表示する
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

            {/* ストリーミング中: 1on1 は characterName、グループは waitingCharacter の名前・色を使う */}
            {sending && (streamingReasoning || (streamingContent !== null && streamingContent.trim().length > 0)) && (() => {
                const streamCharName = waitingCharacter ?? characterName;
                const color = getCharColor(streamCharName);
                return (
                    <div className="flex gap-3 items-start">
                        <CharacterAvatar characterName={streamCharName} imageUrl={getCharImageUrl(streamCharName)} bgClass={color.bg} />
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
                        <CharacterAvatar characterName={charName} imageUrl={getCharImageUrl(charName)} bgClass={color.bg} />
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
