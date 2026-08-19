/**
 * メッセージ一覧表示共通コンポーネント。
 * メッセージのループ表示、自動スクロール、キャラクターごとのカラー設定、考え中インジケーターを管理する。
 */
import { useCallback, useEffect, useRef } from "react";
import type { UIEvent } from "react";
import type { ChatMessage } from "../api";
import { CharacterBubble, CharacterAvatar, CharacterMessageRow, UserBubble, ThinkingBlock, Bubble } from "./ChatBubbles";
import { useHeaderVisibilityOnScroll } from "../hooks/useHeaderVisibilityOnScroll";
import { loadScrollPos, saveScrollPos } from "../lib/sessionSnapshot";

interface Props {
    /** 表示するメッセージ一覧 */
    messages: ChatMessage[];
    /**
     * セッションID。スクロール位置の退避・復元のキーに使う。
     * 未指定なら復元しない（常に最下部へ寄せる従来動作）。
     */
    sessionId?: string;
    /** ユーザ名（表示用） */
    userName: string;
    /** スクロールに応じたヘッダー表示/非表示の通知コールバック。 */
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
    /** メッセージ編集・再生成時のコールバック。
     *  返した Promise が解決するまで再生成ボタンは無効化される（二度押し防止）。 */
    onRetry?: (fromMessageId: string, content: string, imageIds: string[]) => void | Promise<void>;
    /** char_msg_id → log_message_id のマッピング。バブルのログ折りたたみに使用する。 */
    msgLogIds?: Record<string, string>;
    /** char_msg_id → モデル応答完了までの経過時間（ミリ秒）のマッピング。 */
    elapsedMap?: Record<string, number>;
    /** 対面背景の上にバブルが乗る時、視認性のためバブルを半透明にするか。 */
    translucentBubbles?: boolean;
}

/**
 * チャットメッセージのリストを表示し、自動スクロールを制御するコンポーネント。
 */
export default function MessageList({
    messages,
    sessionId,
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
    msgLogIds = {},
    elapsedMap = {},
    translucentBubbles = false,
}: Props) {
    const bottomRef = useRef<HTMLDivElement>(null);
    /** スクロールコンテナ。位置の退避・復元に使う。 */
    const scrollRef = useRef<HTMLDivElement>(null);
    /** 位置復元を済ませたセッションID。セッションごとに 1 回だけ復元するための番人。 */
    const restoredForRef = useRef<string | null>(null);
    /** 退避処理の多重予約を防ぐフラグ（スクロールは連続発火するため rAF で間引く）。 */
    const savePendingRef = useRef(false);
    /** スクロールに応じてヘッダー表示状態を判定する onScroll ハンドラ。 */
    const handleScroll = useHeaderVisibilityOnScroll(onHeaderVisibilityChange);

    /**
     * ヘッダー表示判定に加えて、スクロール位置を退避する onScroll ハンドラ。
     * 1 フレームに 1 回へ間引く（sessionStorage への書き込みが毎イベント走るのを避ける）。
     */
    const handleScrollAndSave = useCallback((e: UIEvent<HTMLDivElement>) => {
        handleScroll(e);
        const el = e.currentTarget;
        if (!sessionId || savePendingRef.current) return;
        savePendingRef.current = true;
        requestAnimationFrame(() => {
            savePendingRef.current = false;
            saveScrollPos(sessionId, {
                scrollTop: el.scrollTop,
                // 最下部付近なら位置ではなく「最新を見ていた」として扱う。
                // 復帰後にメッセージが増えていても最新に追従させたいため。
                atBottom: el.scrollHeight - el.scrollTop - el.clientHeight < 80,
            });
        });
    }, [handleScroll, sessionId]);

    /**
     * キャラクター別配色バブル（cb0〜cb9）を使うかどうか。
     * グループチャット（参加者名リストあり）でのみ true。1on1 はニュートラル面。
     */
    const colored = participantNames.length > 0;

    /**
     * メッセージ追加・ストリーミング・待機中は最下部へスクロールする。
     *
     * セッションを開いた最初の 1 回だけは例外で、退避した位置があればそこへ戻す
     * （過去ログを読んでいる最中にリロードされても、読んでいた場所に戻る）。
     * その初回移動は smooth にしない。長い履歴だと最下部まで延々と流れて見えるため。
     */
    useEffect(() => {
        if (sessionId && restoredForRef.current !== sessionId) {
            // メッセージ描画前に走ると移動先が定まらないので、中身が入るまで待つ。
            if (messages.length === 0) return;
            restoredForRef.current = sessionId;
            const saved = loadScrollPos(sessionId);
            const container = scrollRef.current;
            if (container && saved && !saved.atBottom) {
                container.scrollTop = saved.scrollTop;
                return;
            }
            bottomRef.current?.scrollIntoView({ behavior: "auto" });
            return;
        }
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, sending, streamingContent, waitingCharacter, sessionId]);

    return (
        <div
            ref={scrollRef}
            className={"flex-1 overflow-y-auto overflow-x-hidden" + (translucentBubbles ? " ch-face-to-face-bg" : "")}
            onScroll={handleScrollAndSave}
        >
          {/* pt-16: 浮遊ヘッダー分の上余白。 */}
          <div className="max-w-[760px] mx-auto px-4 sm:px-6 pt-16 pb-6 space-y-3">
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
                        elapsedMs={elapsedMap[msg.id]}
                        // 再ストリームの Promise はそのまま返す（解決まで再生成ボタンを無効化するため）。
                        onRegenerate={onRetry ? () => {
                            const precedingUser = [...messages]
                                .slice(0, idx)
                                .reverse()
                                .find((m) => m.role === "user");
                            if (precedingUser) {
                                return onRetry(precedingUser.id, precedingUser.content, precedingUser.images ?? []);
                            }
                        } : undefined}
                    />
                );
            })}

            {/* ストリーミング中 */}
            {sending && (streamingReasoning || (streamingContent !== null && streamingContent.trim().length > 0)) && (() => {
                const streamCharName = waitingCharacter ?? characterName;
                return (
                    <CharacterMessageRow
                        avatar={<CharacterAvatar characterName={streamCharName} size={28} />}
                        name={streamCharName}
                    >
                        {streamingReasoning && (
                            <div className="mb-1"><ThinkingBlock content={streamingReasoning} streaming /></div>
                        )}
                        {streamingContent !== null && streamingContent.trim().length > 0 && (
                            <Bubble kind="character" colored={colored} characterName={streamCharName}>
                                <span className="whitespace-pre-wrap">{streamingContent}</span>
                                <span className="animate-pulse inline-block ml-0.5 text-ch-accent">▌</span>
                            </Bubble>
                        )}
                    </CharacterMessageRow>
                );
            })()}

            {/* 応答待機インジケーター */}
            {(waitingCharacter || sending) && !streamingReasoning && (streamingContent === null || streamingContent.trim().length === 0) && (() => {
                const charName = waitingCharacter ?? characterName;
                return (
                    <CharacterMessageRow
                        avatar={<CharacterAvatar characterName={charName} size={28} />}
                        name={charName}
                    >
                        <span className="text-ch-t3 text-sm animate-pulse">…</span>
                    </CharacterMessageRow>
                );
            })()}

            <div ref={bottomRef} />
          </div>
        </div>
    );
}
