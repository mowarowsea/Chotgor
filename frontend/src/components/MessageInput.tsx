/**
 * メッセージ入力フォーム共通コンポーネント。
 * テキスト入力（auto-grow）、Ctrl+Enter での送信、画像添付、添付プレビュー機能を提供する。
 * セッション別に入力下書きを localStorage にキャッシュし、セッション切り替え後も復元する。
 */
import { useRef, useState, useEffect } from "react";

interface Props {
    /** セッションID（下書きキャッシュのキー）。省略時はキャッシュしない。 */
    sessionId?: string;
    /** 送信処理中フラグ */
    sending: boolean;
    /** メッセージ送信コールバック。添付された File 配列を含む。 */
    onSend: (content: string, files: File[]) => void;
    /** プレースホルダー文字列 */
    placeholder?: string;
    /** 画像添付を許可するかどうか（デフォルト: true） */
    allowImages?: boolean;
    /** ユーザターンスキップコールバック。指定時はスキップボタンを表示する。 */
    onSkip?: () => void;
}

/** ユーザのメッセージ入力を受け付けるコンポーネント。 */
export default function MessageInput({
    sessionId,
    sending,
    onSend,
    placeholder = "メッセージを入力… (Ctrl+Enter で送信)",
    allowImages = true,
    onSkip,
}: Props) {
    const [input, setInput] = useState("");
    /** 送信前の添付ファイルリスト */
    const [pendingFiles, setPendingFiles] = useState<File[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    /**
     * restoreエフェクト実行直後のsaveエフェクトをスキップするフラグ。
     */
    const skipNextSaveRef = useRef(false);

    /** テキストエリアの高さをコンテンツに合わせて調整する。 */
    const adjustHeight = (el: HTMLTextAreaElement) => {
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 240) + "px";
    };

    /** sessionId が変化したとき、新しいセッションの下書きを復元する。 */
    useEffect(() => {
        if (!sessionId) return;
        const saved = localStorage.getItem(`draft:${sessionId}`) ?? "";
        skipNextSaveRef.current = true;
        setInput(saved);
        requestAnimationFrame(() => {
            if (textareaRef.current) {
                adjustHeight(textareaRef.current);
            }
        });
    }, [sessionId]);

    /** input が変化したとき、下書きを localStorage に保存する。 */
    useEffect(() => {
        if (!sessionId) return;
        if (skipNextSaveRef.current) {
            skipNextSaveRef.current = false;
            return;
        }
        if (input) {
            localStorage.setItem(`draft:${sessionId}`, input);
        } else {
            localStorage.removeItem(`draft:${sessionId}`);
        }
    }, [input, sessionId]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const text = input.trim();
        if (!text || sending) return;

        const files = [...pendingFiles];
        setInput("");
        setPendingFiles([]);
        if (sessionId) localStorage.removeItem(`draft:${sessionId}`);
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
        if (selected.length === 0) return;
        setPendingFiles((prev) => [...prev, ...selected]);
        e.target.value = "";
    };

    const removePendingFile = (idx: number) => {
        setPendingFiles((prev) => prev.filter((_, i) => i !== idx));
    };

    return (
        <form
            onSubmit={handleSubmit}
            className="px-4 sm:px-8 py-3 sm:py-4 flex flex-col gap-2"
            style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}
        >
            {/* 添付画像サムネイルプレビュー */}
            {allowImages && pendingFiles.length > 0 && (
                <div className="flex gap-2 flex-wrap">
                    {pendingFiles.map((file, idx) => (
                        <div key={idx} className="relative group/thumb">
                            <img
                                src={URL.createObjectURL(file)}
                                alt={file.name}
                                className="w-14 h-14 object-cover rounded-lg"
                                style={{ border: "1px solid rgba(255,255,255,0.12)" }}
                            />
                            <button
                                type="button"
                                onClick={() => removePendingFile(idx)}
                                className="absolute -top-1 -right-1 w-4 h-4 bg-ch-s3 hover:bg-ch-s2 rounded-full text-[10px] text-ch-t2 flex items-center justify-center leading-none"
                            >
                                ✕
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* テキストエリアとオーバーレイボタンのコンテナ */}
            <div className="relative">
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    rows={1}
                    disabled={sending}
                    className="w-full bg-ch-s1 text-ch-t1 placeholder-ch-t4 rounded-xl px-4 pt-3 pb-9 text-sm resize-none focus:outline-none disabled:opacity-40 overflow-y-auto"
                    style={{
                        minHeight: "50px",
                        maxHeight: "240px",
                        border: "1px solid rgba(255,255,255,0.08)",
                    }}
                    onFocus={(e) => {
                        e.currentTarget.style.borderColor = "rgba(255,255,255,0.18)";
                        e.currentTarget.style.boxShadow = "0 0 0 3px rgba(255,255,255,0.03)";
                    }}
                    onBlur={(e) => {
                        e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
                        e.currentTarget.style.boxShadow = "none";
                    }}
                />

                {/* テキストエリア内下部のオーバーレイツールバー */}
                <div className="absolute bottom-2 left-2 right-2 flex items-center">
                    {/* ファイル添付ボタン（左端） */}
                    {allowImages && (
                        <>
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
                                className="text-ch-t3 hover:text-ch-t2 disabled:opacity-30 transition-colors p-1 rounded shrink-0"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={16} height={16}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="m18.375 12.739-7.693 7.693a4.5 4.5 0 0 1-6.364-6.364l10.94-10.94A3 3 0 1 1 19.5 7.372L8.552 18.32m.009-.01-.01.01m5.699-9.941-7.81 7.81a1.5 1.5 0 0 0 2.112 2.13" />
                                </svg>
                            </button>
                        </>
                    )}

                    {/* 送信ボタン・スキップボタン（右端） */}
                    <div className="ml-auto flex items-center gap-2">
                        {onSkip && (
                            <button
                                type="button"
                                onClick={onSkip}
                                disabled={sending}
                                className="text-xs text-ch-t3 hover:text-ch-t2 disabled:opacity-30 rounded px-2 py-0.5 transition-colors"
                                style={{ border: "1px solid rgba(255,255,255,0.10)" }}
                            >
                                スキップ
                            </button>
                        )}
                        {/* 送信ボタン — 緑アクセントを使う唯一の構造UI */}
                        <button
                            type="submit"
                            disabled={!input.trim() || sending}
                            className="text-ch-accent-t bg-ch-accent-dim rounded px-3 py-0.5 text-xs font-medium transition-colors disabled:opacity-25"
                            style={{ border: "1px solid rgba(77,140,103,0.35)" }}
                        >
                            送信
                        </button>
                    </div>
                </div>
            </div>
        </form>
    );
}
