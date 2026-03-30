/**
 * 会話エクスポートダイアログ。
 * Reasoning の包含可否・出力先（クリップボード / ファイル）を選択し、
 * フォーマットテンプレートをカスタマイズできる折りたたみセクションを持つ。
 */
import { useState } from "react";
import type { ChatMessage } from "../api";
import {
  useExportFormat,
  buildExportText,
  DEFAULT_CHARACTER_TEMPLATE,
  DEFAULT_USER_TEMPLATE,
} from "../hooks/useExportFormat";

interface Props {
  messages: ChatMessage[];
  userName: string;
  reasoningMap: Record<string, string>;
  sessionTitle?: string;
  onClose: () => void;
}

export default function ExportDialog({
  messages,
  userName,
  reasoningMap,
  sessionTitle,
  onClose,
}: Props) {
  const [includeReasoning, setIncludeReasoning] = useState(false);
  const [destination, setDestination] = useState<"clipboard" | "file">("clipboard");
  const [formatOpen, setFormatOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const { format, updateFormat, resetFormat } = useExportFormat();

  const previewMessages = messages.slice(0, 3);
  const previewText = buildExportText(
    previewMessages,
    reasoningMap,
    userName,
    format,
    includeReasoning,
    sessionTitle,
  );

  const handleExport = async () => {
    const exportText = buildExportText(
      messages,
      reasoningMap,
      userName,
      format,
      includeReasoning,
      sessionTitle,
    );

    if (destination === "clipboard") {
      await navigator.clipboard.writeText(exportText);
      setCopied(true);
      setTimeout(() => {
        setCopied(false);
        onClose();
      }, 1200);
    } else {
      const blob = new Blob([exportText], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${sessionTitle ?? "会話"}.md`;
      a.click();
      URL.revokeObjectURL(url);
      onClose();
    }
  };

  /** テキストエリアの共通スタイル */
  const textareaStyle = {
    border: "1px solid rgba(255,255,255,0.16)",
    outline: "none",
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-ch-s1 rounded-2xl w-full max-w-lg shadow-2xl max-h-[90vh] flex flex-col"
        style={{ border: "1px solid rgba(255,255,255,0.16)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* ヘッダー */}
        <div
          className="flex items-center justify-between px-5 py-4 shrink-0"
          style={{ borderBottom: "1px solid rgba(255,255,255,0.09)" }}
        >
          <h2 className="text-ch-t1 font-medium text-sm">会話をエクスポート</h2>
          <button
            onClick={onClose}
            className="text-ch-t3 hover:text-ch-t2 transition-colors"
            aria-label="閉じる"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" width={16} height={16}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="overflow-y-auto flex-1 px-5 py-4 space-y-4">
          {/* オプション */}
          <div className="space-y-3">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={includeReasoning}
                onChange={(e) => setIncludeReasoning(e.target.checked)}
                className="rounded w-4 h-4"
                style={{ accentColor: "#4d8c67" }}
              />
              <span className="text-ch-t2 text-sm">Reasoning を含む</span>
            </label>
            <div className="flex gap-5">
              {(["clipboard", "file"] as const).map((dest) => (
                <label key={dest} className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="radio"
                    name="export-destination"
                    value={dest}
                    checked={destination === dest}
                    onChange={() => setDestination(dest)}
                    className="w-4 h-4"
                    style={{ accentColor: "#4d8c67" }}
                  />
                  <span className="text-ch-t2 text-sm">
                    {dest === "clipboard" ? "クリップボード" : "ファイル (.md)"}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* フォーマットカスタマイズ（折りたたみ） */}
          <div className="rounded-xl overflow-hidden" style={{ border: "1px solid rgba(255,255,255,0.12)" }}>
            <button
              className="w-full flex items-center gap-2 px-4 py-2.5 text-ch-t3 hover:text-ch-t2 hover:bg-ch-s2/60 transition-colors text-left text-xs"
              onClick={() => setFormatOpen((o) => !o)}
            >
              <span className="text-[10px] opacity-60">{formatOpen ? "▼" : "▶"}</span>
              <span>フォーマットをカスタマイズ</span>
            </button>
            {formatOpen && (
              <div className="px-4 py-3 space-y-3" style={{ borderTop: "1px solid rgba(255,255,255,0.09)" }}>
                <p className="text-ch-t3 text-xs leading-relaxed">
                  使用できる変数:{" "}
                  {[
                    "{character_name}",
                    "{user_name}",
                    "{content}",
                    "{reasoning}",
                    "{timestamp}",
                  ].map((v) => (
                    <code
                      key={v}
                      className="bg-ch-s3 text-ch-accent-t rounded px-1 py-0.5 font-mono text-[0.8em] mr-1"
                    >
                      {v}
                    </code>
                  ))}
                </p>
                <div className="space-y-1">
                  <label className="text-ch-t3 text-xs block">キャラクターターン</label>
                  <textarea
                    value={format.characterTemplate}
                    onChange={(e) => updateFormat({ characterTemplate: e.target.value })}
                    rows={5}
                    spellCheck={false}
                    className="w-full bg-ch-bg text-ch-t1 text-xs font-mono rounded-lg px-3 py-2 resize-y focus:outline-none"
                    style={textareaStyle}
                    onFocus={(e) => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.35)"; }}
                    onBlur={(e) => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.16)"; }}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-ch-t3 text-xs block">ユーザーターン</label>
                  <textarea
                    value={format.userTemplate}
                    onChange={(e) => updateFormat({ userTemplate: e.target.value })}
                    rows={3}
                    spellCheck={false}
                    className="w-full bg-ch-bg text-ch-t1 text-xs font-mono rounded-lg px-3 py-2 resize-y focus:outline-none"
                    style={textareaStyle}
                    onFocus={(e) => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.35)"; }}
                    onBlur={(e) => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.16)"; }}
                  />
                </div>
                <button
                  onClick={resetFormat}
                  className="text-ch-t3 hover:text-ch-t2 text-xs px-2 py-1 rounded transition-colors"
                  style={{ border: "1px solid rgba(255,255,255,0.12)" }}
                >
                  デフォルトに戻す
                </button>
              </div>
            )}
          </div>

          {/* プレビュー */}
          <div className="space-y-1.5">
            <p className="text-ch-t3 text-xs">
              プレビュー（先頭 {Math.min(3, messages.length)} 件）
            </p>
            <pre
              className="bg-ch-bg rounded-xl px-4 py-3 text-ch-t2 text-xs font-mono whitespace-pre-wrap max-h-40 overflow-y-auto leading-relaxed"
              style={{ border: "1px solid rgba(255,255,255,0.10)" }}
            >
              {previewText || "（メッセージなし）"}
            </pre>
          </div>
        </div>

        {/* フッター */}
        <div
          className="px-5 py-4 flex justify-end gap-2 shrink-0"
          style={{ borderTop: "1px solid rgba(255,255,255,0.09)" }}
        >
          <button
            onClick={onClose}
            className="text-ch-t3 hover:text-ch-t2 text-sm px-4 py-2 rounded-lg transition-colors"
          >
            キャンセル
          </button>
          <button
            onClick={handleExport}
            disabled={messages.length === 0 || copied}
            className="text-ch-accent-t text-sm px-4 py-2 rounded-lg transition-colors flex items-center gap-1.5 disabled:opacity-30"
            style={{
              background: "rgba(22,22,22,0.9)",
              border: "1px solid rgba(255,255,255,0.32)",
            }}
          >
            {copied ? (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" width={14} height={14}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                </svg>
                コピー済み
              </>
            ) : destination === "clipboard" ? (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={14} height={14}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0 0 13.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 0 1-.75.75H9a.75.75 0 0 1-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 0 1-2.25 2.25H6.75A2.25 2.25 0 0 1 4.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 0 1 1.927-.184" />
                </svg>
                クリップボードへコピー
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width={14} height={14}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>
                ダウンロード
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// デフォルトテンプレートを再エクスポート（他のファイルが必要な場合用）
export { DEFAULT_CHARACTER_TEMPLATE, DEFAULT_USER_TEMPLATE };
