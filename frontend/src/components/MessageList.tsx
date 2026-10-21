/**
 * メッセージ一覧表示共通コンポーネント。
 * メッセージのループ表示、自動スクロール、キャラクターごとのカラー設定、考え中インジケーターを管理する。
 */
import { useCallback, useEffect, useRef } from "react";
import type { UIEvent } from "react";
import type { Attachment, ChatMessage } from "../api";
import { CharacterBubble, CharacterAvatar, CharacterMessageRow, UserBubble, ThinkingBlock, Bubble } from "./ChatBubbles";
import { useHeaderVisibilityOnScroll } from "../hooks/useHeaderVisibilityOnScroll";
import { scrollAnchorToTop } from "../lib/scrollAnchor";
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
    /** ユーザ発話の編集コールバック（以降を削除して再送する）。
     *  返した Promise が解決するまで再生成ボタンは無効化される（二度押し防止）。 */
    onRetry?: (fromMessageId: string, content: string, attachments: Attachment[]) => void | Promise<void>;
    /**
     * キャラクター応答の引き直し（再生成）コールバック。
     * `replaced` には引き直しで置き換わる旧応答（起点ユーザ発話以降）を渡す。
     * 未指定なら onRetry へフォールバックする（旧挙動: 先に消してから再送）。
     */
    onRegenerate?: (
        fromMessageId: string,
        content: string,
        attachments: Attachment[],
        replaced: ChatMessage[],
    ) => void | Promise<void>;
    /** 末尾ユーザメッセージの削除コールバック。渡された時だけ末尾バブルにゴミ箱を出す。 */
    onDeleteMessage?: (messageId: string) => void | Promise<void>;
    /**
     * 最新ターンの応答を index 番目（1始まり）の枝へ切り替えるコールバック。
     * 渡されていて、末尾のキャラ応答に variant_count >= 2 があるときだけ ◀ ▶ を出す。
     */
    onSwitchVariant?: (anchorMessageId: string, index: number) => void | Promise<void>;
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
    onRegenerate,
    onDeleteMessage,
    onSwitchVariant,
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
    /** 直前の描画時点の sending。送信中→完了の切り替わりを検出するために持つ。 */
    const prevSendingRef = useRef(sending);
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
     * 例外は 2 つで、どちらも最後のユーザ発話を画面上端へ寄せる（長い応答を頭から読めるように）:
     *   - 応答が出揃った瞬間（送信中→完了）。smooth で動かす。
     *   - セッションを開いた最初の 1 回（リロード・サイドバーからの切替）。ぱっと出す。
     *     ただし退避した位置があればそこへ戻す（過去ログを読んでいる最中にリロードされても、
     *     読んでいた場所に戻る）。
     * 応答が短ければスクロール量が足りずに最下部のまま止まるので、従来と同じ見た目になる。
     *
     * 依存は messages そのものではなく末尾 id にする。スナップショット描画のあとの
     * API 再取得など、中身が同じ配列の差し替えで最下部へ飛ばされないようにするため。
     */
    const lastMessageId = messages.length > 0 ? messages[messages.length - 1].id : null;
    /** 表示中メッセージの持ち主。切替直後は前のセッションの messages が残っている。 */
    const messagesSessionId = messages.length > 0 ? messages[messages.length - 1].session_id : null;
    useEffect(() => {
        const justFinished = prevSendingRef.current && !sending;
        prevSendingRef.current = sending;
        const container = scrollRef.current;
        const lastUserBubble = () => {
            const userBubbles = container?.querySelectorAll<HTMLElement>("[data-user-bubble]");
            return userBubbles?.[userBubbles.length - 1] ?? null;
        };
        if (sessionId && restoredForRef.current !== sessionId) {
            // このセッションのメッセージが描画されるまで待つ（移動先が定まらないため）。
            if (messagesSessionId !== sessionId) return;
            restoredForRef.current = sessionId;
            // 新規セッションの初回送信で楽観バブルが入った場合など。生成中は最下部追従に任せる。
            if (sending) {
                bottomRef.current?.scrollIntoView({ behavior: "auto" });
                return;
            }
            const saved = loadScrollPos(sessionId);
            if (container && saved && !saved.atBottom) {
                container.scrollTop = saved.scrollTop;
                return;
            }
            const anchor = lastUserBubble();
            if (container && anchor) {
                scrollAnchorToTop(container, anchor, "auto");
                return;
            }
            bottomRef.current?.scrollIntoView({ behavior: "auto" });
            return;
        }
        const anchor = justFinished ? lastUserBubble() : null;
        if (container && anchor) {
            scrollAnchorToTop(container, anchor, "smooth");
            return;
        }
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [lastMessageId, messagesSessionId, sending, streamingContent, waitingCharacter, sessionId]);

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
                    // ゴミ箱はセッション末尾のユーザ発話にだけ出す。削除 API は「以降まとめて」しか
                    // 持たないので、後続のある発話に出すと巻き添えで消えてしまうため。
                    const isTail = idx === messages.length - 1;
                    return (
                        <UserBubble
                            key={msg.id}
                            content={msg.content}
                            userName={userName}
                            attachments={msg.attachments}
                            sending={sending}
                            onEdit={onRetry ? (newContent) => onRetry(msg.id, newContent, msg.attachments ?? []) : undefined}
                            onDelete={
                                onDeleteMessage && isTail && !sending
                                    ? () => onDeleteMessage(msg.id)
                                    : undefined
                            }
                        />
                    );
                }

                const charName = msg.character_name ?? characterName;
                // 枝ナビは最新ターン限定なので、セッション末尾のキャラ応答にだけ出す。
                // 次の発話が積まれた時点で、古い応答の variant_* は残っていても表示しない。
                const anchorId = msg.variant_anchor_id;
                const showVariants =
                    !!onSwitchVariant && !!anchorId && idx === messages.length - 1;
                const variantIndex = msg.variant_index ?? 1;

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
                        variantIndex={showVariants ? variantIndex : undefined}
                        variantCount={showVariants ? msg.variant_count : undefined}
                        onPrevVariant={showVariants ? () => onSwitchVariant(anchorId, variantIndex - 1) : undefined}
                        onNextVariant={showVariants ? () => onSwitchVariant(anchorId, variantIndex + 1) : undefined}
                        // 再ストリームの Promise はそのまま返す（解決まで再生成ボタンを無効化するため）。
                        onRegenerate={onRegenerate || onRetry ? () => {
                            let userIdx = -1;
                            for (let i = idx - 1; i >= 0; i--) {
                                if (messages[i].role === "user") { userIdx = i; break; }
                            }
                            if (userIdx < 0) return;
                            const precedingUser = messages[userIdx];
                            if (onRegenerate) {
                                // 置き換わる旧応答（起点ユーザ発話以降）も渡す。引き直しに
                                // 失敗したとき、呼び出し側がこれを画面へ戻すために使う。
                                return onRegenerate(
                                    precedingUser.id,
                                    precedingUser.content,
                                    precedingUser.attachments ?? [],
                                    messages.slice(userIdx + 1),
                                );
                            }
                            if (onRetry) {
                                return onRetry(precedingUser.id, precedingUser.content, precedingUser.attachments ?? []);
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
