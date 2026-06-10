/**
 * バブル操作ボタン群 — コピー / 再生成 / 破棄 / ユーザー発話操作バー。
 * 1on1・グループ・シナリオの全モードで同じ見た目を共有する共通部品。
 */
import { useCallback, useState } from "react";

/** テキストをクリップボードにコピーし、完了時に一時的にチェックマークを表示するボタン。 */
export function CopyButton({ text, className = "" }: { text: string; className?: string }) {
  const [done, setDone] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const el = document.createElement("textarea");
      el.value = text;
      el.style.cssText = "position:fixed;opacity:0;pointer-events:none";
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
    }
    setDone(true);
    setTimeout(() => setDone(false), 1500);
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      title="コピー"
      className={`opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-ch-t3 hover:text-ch-t2 transition-all p-1 rounded shrink-0 ${className}`}
    >
      {done ? (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" width={12} height={12}>
          <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={12} height={12}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0 0 13.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 0 1-.75.75H9a.75.75 0 0 1-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 0 1-2.25 2.25H6.75A2.25 2.25 0 0 1 4.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 0 1 1.927-.184" />
        </svg>
      )}
    </button>
  );
}

/**
 * バブル下部の再生成（↺）ボタン。
 *
 * 誤クリック防止のため、呼び出し側で `ml-auto` 等によりバブルの右端へ
 * 寄せて使うことを想定している。1on1 / グループ / シナリオの全モードで
 * 同じ見た目を共有するための共通部品。
 */
export function RegenerateButton({
  onClick,
  title = "再生成",
  className = "",
}: {
  /** クリック時のコールバック。 */
  onClick: () => void;
  /** ホバー時のツールチップ。 */
  title?: string;
  /** 追加クラス（右端寄せの `ml-auto` 等）。 */
  className?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-ch-t3 hover:text-ch-t2 text-xs transition-all p-1 rounded ${className}`}
    >
      ↺
    </button>
  );
}

/**
 * バブル下部の応答破棄（🗑）ボタン。
 *
 * シナリオモードで「直前の GM 応答を捨てて、ユーザリクエスト待ち状態へ戻したい」
 * 用途で使う。`RegenerateButton` と異なり、削除後に再ストリームは行わない。
 * 誤クリック防止のため、呼び出し側で `ml-auto` 等によりバブル右端へ寄せて使う。
 */
export function DiscardButton({
  onClick,
  title = "この応答を破棄",
  className = "",
}: {
  /** クリック時のコールバック。 */
  onClick: () => void;
  /** ホバー時のツールチップ。 */
  title?: string;
  /** 追加クラス（右端寄せの `ml-auto` 等）。 */
  className?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-ch-t3 hover:text-red-500 transition-all p-1 rounded ${className}`}
      aria-label="この応答を破棄"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        strokeWidth={1.5}
        stroke="currentColor"
        width={13}
        height={13}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"
        />
      </svg>
    </button>
  );
}

/**
 * ユーザー発話バブル下部の操作バー（コピー / 編集）。
 *
 * バブルの直下に置き、親の `items-end` により右寄せされる前提。
 * 1on1 / シナリオのユーザーバブルで見た目・並びを共有する（DRY）。
 */
export function UserMessageActions({
  copyText,
  onEdit,
}: {
  /** コピー対象テキスト。 */
  copyText: string;
  /** 編集開始コールバック（無指定で編集ボタン非表示）。 */
  onEdit?: () => void;
}) {
  return (
    <div className="flex items-center gap-0.5 mt-0.5">
      <CopyButton text={copyText} />
      {onEdit && (
        <button
          onClick={onEdit}
          title="編集"
          className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 text-ch-t3 hover:text-ch-t2 transition-all p-1 rounded shrink-0"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={12} height={12}>
            <path strokeLinecap="round" strokeLinejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10" />
          </svg>
        </button>
      )}
    </div>
  );
}
