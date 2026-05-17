/**
 * セッション別のメッセージ下書きを localStorage に永続化する hook。
 *
 * `draft:{sessionId}` をキーに、入力途中のテキストを保存・復元する。
 * セッション切替時（sessionId が変わったとき）に該当セッションの下書きを復元し、
 * 入力が変わるたびに保存する。空文字列はキーを削除する。
 *
 * 1on1 / シナリオチャットなど複数の入力欄から再利用する目的で抽出された。
 */
import { useEffect, useRef, useState } from "react";

/**
 * セッション別下書きを管理する hook。
 *
 * @param sessionId 現在のセッション ID。undefined を渡すと永続化しない（in-memory のみ）。
 * @returns `[input, setInput]` — useState と同じインターフェース。
 *          `setInput("")` で localStorage 側も削除される（送信完了後の自然なクリア）。
 */
export function useDraft(
    sessionId: string | undefined,
): [string, React.Dispatch<React.SetStateAction<string>>] {
    const [input, setInput] = useState("");
    /**
     * restore（sessionId 変化に伴う localStorage → state 反映）直後の save エフェクトを
     * 1 回だけスキップするためのフラグ。これがないと、復元したテキストでまた save が走り
     * 「読み込んだそばから書き込む」無駄が発生する（実害は薄いが、別タブとの競合や
     * skipNextSave を期待する呼び出し側で意図せぬ動きの元になる）。
     */
    const skipNextSaveRef = useRef(false);

    // sessionId が変わったら、新しいセッションの下書きを復元する。
    useEffect(() => {
        if (!sessionId) return;
        const saved = localStorage.getItem(`draft:${sessionId}`) ?? "";
        skipNextSaveRef.current = true;
        setInput(saved);
    }, [sessionId]);

    // input 変化のたびに保存。空文字なら削除。
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

    return [input, setInput];
}
