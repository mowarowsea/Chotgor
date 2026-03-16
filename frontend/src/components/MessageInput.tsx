/**
 * メッセージ入力フォーム共通コンポーネント。
 * テキスト入力（auto-grow）、Shift+Enter での送信、画像添付、添付プレビュー機能を提供する。
 * ファイル添付ボタンと送信ボタンはテキストエリア内下部に配置する。
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
    /** プレースホルダー文字列（デフォルト: "メッセージを入力… (Shift+Enter で送信)"） */
    placeholder?: string;
    /** 画像添付を許可するかどうか（デフォルト: true） */
    allowImages?: boolean;
}

/**
 * ユーザのメッセージ入力を受け付けるコンポーネント。
 */
export default function MessageInput({
    sessionId,
    sending,
    onSend,
    placeholder = "メッセージを入力… (Shift+Enter で送信)",
    allowImages = true,
}: Props) {
    const [input, setInput] = useState("");
    /** 送信前の添付ファイルリスト */
    const [pendingFiles, setPendingFiles] = useState<File[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    /**
     * restoreエフェクト実行直後のsaveエフェクトをスキップするフラグ。
     * restoreがinputを上書きする前にsaveが空文字でdraftを削除するのを防ぐ。
     * React StrictModeの二重実行にも対応するためrefで管理する。
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
        // 直後のsaveエフェクトが空inputでdraftを削除しないようスキップフラグを立てる
        skipNextSaveRef.current = true;
        setInput(saved);
        // 高さの復元は DOM 更新後に行う
        requestAnimationFrame(() => {
            if (textareaRef.current) {
                adjustHeight(textareaRef.current);
            }
        });
    }, [sessionId]);

    /** input が変化したとき、下書きを localStorage に保存する。 */
    useEffect(() => {
        if (!sessionId) return;
        // restoreエフェクトの直後はスキップ（古いinputでdraftを上書きするのを防ぐ）
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
        // 高さをリセット
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
        <form
            onSubmit={handleSubmit}
            className="border-t border-zinc-800 px-3 sm:px-6 py-3 sm:py-4 flex flex-col gap-2"
        >
            {/* 添付画像サムネイルプレビュー */}
            {allowImages && pendingFiles.length > 0 && (
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
                    className="w-full bg-zinc-800 text-zinc-100 placeholder-zinc-500 rounded-xl px-4 pt-3 pb-10 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50 overflow-y-auto"
                    style={{ minHeight: "52px", maxHeight: "240px" }}
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
                                className="text-zinc-500 hover:text-zinc-300 disabled:opacity-40 transition-colors p-1.5 rounded-lg hover:bg-zinc-700 shrink-0"
                            >
                                {/* Heroicons: paper-clip */}
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={18} height={18}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="m18.375 12.739-7.693 7.693a4.5 4.5 0 0 1-6.364-6.364l10.94-10.94A3 3 0 1 1 19.5 7.372L8.552 18.32m.009-.01-.01.01m5.699-9.941-7.81 7.81a1.5 1.5 0 0 0 2.112 2.13" />
                                </svg>
                            </button>
                        </>
                    )}

                    {/* 送信ボタン（右端） */}
                    <button
                        type="submit"
                        disabled={!input.trim() || sending}
                        className="ml-auto bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-lg px-3 py-1 text-xs font-medium transition-colors"
                    >
                        送信
                    </button>
                </div>
            </div>
        </form>
    );
}
