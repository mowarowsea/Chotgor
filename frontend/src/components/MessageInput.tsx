/**
 * メッセージ入力フォーム共通コンポーネント。
 * テキスト入力（auto-grow）、Ctrl+Enter での送信、添付（画像・音声）、添付プレビュー機能を提供する。
 * セッション別に入力下書きを localStorage にキャッシュし、セッション切り替え後も復元する。
 */
import { useEffect, useRef, useState } from "react";
import { useDraft } from "../hooks/useDraft";
import { useEditingLocked } from "../hooks/useEditingLock";
import {
    acceptAttribute,
    attachmentButtonTitle,
    attachmentKind,
    rejectionReason,
    type AttachmentKind,
} from "../lib/attachments";

interface Props {
    /** セッションID（下書きキャッシュのキー）。省略時はキャッシュしない。 */
    sessionId?: string;
    /** 送信処理中フラグ */
    sending: boolean;
    /** メッセージ送信コールバック。添付された File 配列を含む。 */
    onSend: (content: string, files: File[]) => void;
    /** プレースホルダー文字列 */
    placeholder?: string;
    /** 添付を許可するかどうか（デフォルト: true） */
    allowAttachments?: boolean;
    /**
     * 選択中プリセットが受け取れる添付種別（/v1/models の attachment_kinds）。
     * FileDialog の accept と選択後の検査の根拠。省略時は画像のみ。
     */
    attachmentKinds?: AttachmentKind[];
    /** ユーザターンスキップコールバック。指定時はスキップボタンを表示する。 */
    onSkip?: () => void;
    /**
     * 空文字の送信を許可するかどうか（デフォルト: false）。
     * シナリオの「空欄送信で GM が無言のまま物語を進める」モード用。
     */
    allowEmptySend?: boolean;
    /**
     * 送信ボタンの直前に挿入される追加ツール。
     * シナリオの「宛先 PC 切替トグル」など、ページ固有の補助 UI を差し込むため。
     */
    extraTools?: React.ReactNode;
}

/** ユーザのメッセージ入力を受け付けるコンポーネント。 */
export default function MessageInput({
    sessionId,
    sending,
    onSend,
    placeholder = "メッセージを入力… (Ctrl+Enter で送信)",
    allowAttachments = true,
    attachmentKinds = ["image"],
    onSkip,
    allowEmptySend = false,
    extraTools,
}: Props) {
    // 下書きの localStorage 連携は useDraft hook に集約済み。
    // setInput("") を呼ぶと hook 内の useEffect が走り、localStorage の該当キーも削除される。
    const [input, setInput] = useDraft(sessionId);
    /**
     * バブルのインライン編集中フラグ。
     * 編集中の誤送信は「以降のターンを巻き戻して再ストリーム」を走らせ、
     * 編集内容ごと失わせるため、入力・送信をまとめて止める。
     */
    const editingLocked = useEditingLocked();
    /** 入力を受け付けない状態（送信処理中 or バブル編集中）。 */
    const disabled = sending || editingLocked;
    /** 送信前の添付ファイルリスト */
    const [pendingFiles, setPendingFiles] = useState<File[]>([]);
    /** 添付が弾かれた理由（非対応形式・非対応プロバイダー）。次の選択で消える。 */
    const [attachmentError, setAttachmentError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    /**
     * テキストエリアの高さをコンテンツに合わせて調整する。
     * 内容が最大高さ（240px）に収まる間は overflow を hidden にして
     * 不要なスクロールバーを出さない。超えたときだけ auto で表示する。
     * 右下にツールボタンを重ねて表示するため、paddingBottom で
     * 入力量＋1行分のスクロール余裕を確保している（INPUT_BOTTOM_PAD）。
     */
    const adjustHeight = (el: HTMLTextAreaElement) => {
        el.style.height = "auto";
        const full = el.scrollHeight;
        el.style.height = Math.min(full, 240) + "px";
        el.style.overflowY = full > 240 ? "auto" : "hidden";
    };

    // 復元された下書きに合わせて textarea の高さを調整する。
    // useDraft が input を更新したフレームでは textareaRef.current.scrollHeight が
    // まだ古いままなので、次フレームで再計算する。
    useEffect(() => {
        if (textareaRef.current) {
            adjustHeight(textareaRef.current);
        }
    }, [input]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const text = input.trim();
        if (disabled) return;
        if (!text && !allowEmptySend) return;

        const files = [...pendingFiles];
        // setInput("") は useDraft の useEffect 経由で localStorage キーも削除する。
        setInput("");
        setPendingFiles([]);
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
        }
        onSend(text, files);
    };

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(e.target.value);
        adjustHeight(e.target);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && e.ctrlKey) {
            e.preventDefault();
            handleSubmit(e as unknown as React.FormEvent);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selected = Array.from(e.target.files ?? []);
        e.target.value = "";
        if (selected.length === 0) return;
        // accept をすり抜けた分をここで弾く（スマホの FileDialog は accept を
        // 尊重しないことがある）。黙って捨てず、弾いた理由を出す。
        const rejected = selected
            .map((f) => rejectionReason(f, attachmentKinds))
            .filter((r): r is string => r !== null);
        const accepted = selected.filter((f) => rejectionReason(f, attachmentKinds) === null);
        setAttachmentError(rejected.length > 0 ? rejected.join(" / ") : null);
        if (accepted.length > 0) {
            setPendingFiles((prev) => [...prev, ...accepted]);
        }
    };

    const removePendingFile = (idx: number) => {
        setPendingFiles((prev) => prev.filter((_, i) => i !== idx));
    };

    // プリセットを切り替えて渡せなくなった添付は持ち越さない
    // （選び直しは効くが、渡らないものを抱えたまま送れてしまうのを防ぐ）。
    useEffect(() => {
        setPendingFiles((prev) => {
            const kept = prev.filter((f) => rejectionReason(f, attachmentKinds) === null);
            return kept.length === prev.length ? prev : kept;
        });
        // attachmentKinds は親が毎回新しい配列を渡しうるので中身で比較する
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [attachmentKinds.join(",")]);

    /** 送信可能かどうか。送信ボタンの配色に使う（空送信許可時は常に点灯）。 */
    const canSend = !disabled && (!!input.trim() || allowEmptySend);

    /** 入力欄が空のときに出す擬似 placeholder。編集中はその旨を伝える。 */
    const placeholderText = editingLocked
        ? "メッセージを編集中は送信できません"
        : placeholder;

    return (
        <form
            onSubmit={handleSubmit}
            style={{ borderTop: "1px solid var(--ch-sep)" }}
        >
            {/* 中央寄せ・最大幅 760px のコンテナ */}
            <div className="max-w-[760px] mx-auto px-4 sm:px-6 pt-0.5 pb-3.5 flex flex-col gap-2">
                {/* 添付が弾かれた理由 */}
                {allowAttachments && attachmentError && (
                    <div className="text-[11px] text-red-500">{attachmentError}</div>
                )}

                {/* 添付プレビュー: 画像はサムネ、音声はファイル名チップ＋プレイヤー
                    （音声を createObjectURL で <img> に食わせると壊れるため描き分ける）。 */}
                {allowAttachments && pendingFiles.length > 0 && (
                    <div className="flex gap-2 flex-wrap items-start">
                        {pendingFiles.map((file, idx) => (
                            <div key={idx} className="relative group/thumb">
                                {attachmentKind(file.type) === "audio" ? (
                                    <div
                                        className="flex flex-col gap-1 rounded-md px-2 py-1.5 max-w-[240px]"
                                        style={{ border: "1px solid var(--ch-sep2)" }}
                                    >
                                        <span className="text-[11px] text-ch-t2 truncate" title={file.name}>
                                            ♪ {file.name}
                                        </span>
                                        <audio
                                            controls
                                            preload="metadata"
                                            src={URL.createObjectURL(file)}
                                            className="h-8 max-w-full"
                                        />
                                    </div>
                                ) : (
                                    <img
                                        src={URL.createObjectURL(file)}
                                        alt={file.name}
                                        className="w-14 h-11 object-cover rounded-md"
                                        style={{ border: "1px solid var(--ch-sep2)" }}
                                    />
                                )}
                                <button
                                    type="button"
                                    onClick={() => removePendingFile(idx)}
                                    className="absolute -top-1 -right-1 w-4 h-4 bg-ch-s3 hover:bg-ch-s2 rounded-full text-[10px] text-ch-t2 flex items-center justify-center leading-none"
                                    style={{ border: "1px solid var(--ch-sep2)" }}
                                >
                                    ✕
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {/* 入力行: 下線スタイルのテキストエリア + ツールボタン（右下に絶対配置で重ねる） */}
                <div className="relative">
                    <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={handleChange}
                        onKeyDown={handleKeyDown}
                        // 標準の placeholder は paddingBottom（ツールボタン用の余白）の影響を受けて
                        // 2行目もボタン上に描画されてしまう。要件「2行目はボタンと同じ行に出したい」を
                        // 満たすため placeholder 属性は使わず、下で擬似 placeholder を被せる。
                        rows={1}
                        disabled={disabled}
                        className="w-full block bg-transparent text-ch-t1 text-sm resize-none focus:outline-none disabled:opacity-40 pt-1.5 leading-relaxed"
                        style={{
                            minHeight: "32px",
                            maxHeight: "240px",
                            // 初期は hidden。内容が 240px を超えたら adjustHeight が auto にする。
                            overflowY: "hidden",
                            borderBottom: "1px solid var(--ch-sep)",
                            transition: "border-color .2s",
                            // 入力末尾がツールボタンに被って隠れないよう、1行分のスクロール余白を確保。
                            paddingBottom: "28px",
                        }}
                        onFocus={(e) => {
                            e.currentTarget.style.borderBottomColor = "var(--ch-accent)";
                        }}
                        onBlur={(e) => {
                            e.currentTarget.style.borderBottomColor = "var(--ch-sep)";
                        }}
                    />

                    {/* 擬似 placeholder: textarea が空のときだけ表示する。
                        textarea の paddingBottom（ツールボタン用の余白）を無視させたいので、
                        標準の placeholder 属性ではなく独立した overlay にしている。
                        2行目がツールボタンと同じ行に重なる位置に出るのが意図した挙動。 */}
                    {!input && (
                        <div
                            aria-hidden
                            className="absolute inset-x-0 top-0 text-ch-t3 text-sm leading-relaxed pt-1.5 pointer-events-none"
                            style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
                        >
                            {placeholderText}
                        </div>
                    )}

                    {/* ツールボタン群（テキストエリア右下に重ねる） */}
                    <div className="absolute right-0 bottom-1.5 flex items-center gap-1.5 pointer-events-auto">
                        {allowAttachments && (
                            <>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept={acceptAttribute(attachmentKinds)}
                                    multiple
                                    className="hidden"
                                    onChange={handleFileChange}
                                />
                                <button
                                    type="button"
                                    onClick={() => fileInputRef.current?.click()}
                                    disabled={disabled}
                                    title={attachmentButtonTitle(attachmentKinds)}
                                    className="text-ch-t3 hover:text-ch-t2 disabled:opacity-30 transition-colors p-0.5 rounded"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={17} height={17}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="m18.375 12.739-7.693 7.693a4.5 4.5 0 0 1-6.364-6.364l10.94-10.94A3 3 0 1 1 19.5 7.372L8.552 18.32m.009-.01-.01.01m5.699-9.941-7.81 7.81a1.5 1.5 0 0 0 2.112 2.13" />
                                    </svg>
                                </button>
                            </>
                        )}
                        {onSkip && (
                            <button
                                type="button"
                                onClick={onSkip}
                                disabled={disabled}
                                className="text-xs text-ch-t3 hover:text-ch-t2 disabled:opacity-30 rounded px-1.5 py-0.5 transition-colors"
                            >
                                スキップ
                            </button>
                        )}
                        {extraTools}
                        {/* 送信ボタン: 入力がある時のみアクセント色で点灯する */}
                        <button
                            type="submit"
                            disabled={!canSend}
                            title="送信 (Ctrl+Enter)"
                            className="p-0.5 rounded transition-colors disabled:cursor-not-allowed"
                            style={{ color: canSend ? "var(--ch-accent)" : "var(--ch-t3)" }}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" width={18} height={18}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M12 19V5M5 12l7-7 7 7" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </form>
    );
}
