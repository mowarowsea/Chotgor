/**
 * MessageActionBar — バブル下部の操作バー（コピー / 経過時間 / ログ / 再生成 / 破棄）。
 */
import { useState } from "react";

import { fetchLogEntry } from "../../api";
import type { LogEntry } from "../../api";
import { CopyButton, DiscardButton, RegenerateButton } from "./buttons";
import { RawLogModal, TAG_COLORS, ToolCallRow } from "./logViewer";

/**
 * 経過時間（ミリ秒）を人間向けの短い文字列にフォーマットする。
 *
 * - 60秒未満: `12.3s` 形式（小数1桁）
 * - 60秒以上: `1m05s` 形式（分・秒、秒は2桁ゼロ詰め）
 */
function formatElapsed(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "";
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return `${m}m${String(s).padStart(2, "0")}s`;
}

/**
 * バブル下部の操作バー。
 *
 * コピー / ログ折りたたみ（1on1 のみ）/ 再生成 を 1 行に並べる共通部品。
 * 1on1・グループ・シナリオの全モードで同じ並び・見た目を共有する（DRY）。
 * 再生成は誤クリック防止のため `ml-auto` で右端（バブル右端）へ寄せる。
 *
 * `logMessageId` が渡されたときのみログ折りたたみを表示する
 * （CHOTGOR_DEBUG=1 時の 1on1 チャット用）。トリガーは操作行内に置き、
 * 展開コンテンツは操作行の下にぶら下げる。
 */
export function MessageActionBar({
  copyText,
  onRegenerate,
  regenerateTitle = "再生成",
  onDiscard,
  discardTitle = "この応答を破棄",
  logMessageId,
  elapsedMs,
}: {
  /** コピーボタンがコピーするテキスト。 */
  copyText: string;
  /** 再生成コールバック（無指定で再生成ボタン非表示）。 */
  onRegenerate?: () => void;
  /** 再生成ボタンのツールチップ。 */
  regenerateTitle?: string;
  /** 応答破棄コールバック（無指定で破棄ボタン非表示）。再ストリームは行わない。 */
  onDiscard?: () => void;
  /** 破棄ボタンのツールチップ。 */
  discardTitle?: string;
  /** デバッグログフォルダ名（8桁hex）。指定時のみログ折りたたみを表示する。 */
  logMessageId?: string;
  /** モデルリクエスト〜応答完了までの経過時間（ミリ秒）。指定時のみ表示する。 */
  elapsedMs?: number;
}) {
  const [logExpanded, setLogExpanded] = useState(false);
  const [entry, setEntry] = useState<LogEntry | null>(null);
  const [loading, setLoading] = useState(false);
  const [rawModal, setRawModal] = useState<string | null>(null);

  /** ログ折りたたみのトグル。初回展開時にログを遅延取得する。 */
  const handleLogToggle = async () => {
    if (!logMessageId) return;
    if (!logExpanded && !entry && !loading) {
      setLoading(true);
      try {
        setEntry(await fetchLogEntry(logMessageId));
      } catch {
        // ロード失敗時はエントリなしのまま表示する
      } finally {
        setLoading(false);
      }
    }
    setLogExpanded((e) => !e);
  };

  /** ▼ログ展開時に表示するツール呼び出し一覧。
   *
   * メイン行（chat/scenario 等）の tool_calls は `entry.attempts[].tool_calls` に格納される
   * （再生成複数試行のため）。バブル本体は「最新試行のレスポンス」のみを表示しているので、
   * ログも `attempts` 末尾 = 最新試行ぶんだけ表示する（過去試行の重複表示を避ける）。
   * 非メイン行のみのエントリ（chronicle/forget 等）は `attempts` が空なので top-level を使う。
   */
  const latestAttempt =
    entry && entry.attempts && entry.attempts.length > 0
      ? entry.attempts[entry.attempts.length - 1]
      : null;

  const toolCalls = latestAttempt
    ? latestAttempt.tool_calls
    : (entry?.tool_calls ?? []);

  /** ▼ログ展開時に表示する警告一覧。tool_calls と同様に最新試行のみ。 */
  const warnings = latestAttempt
    ? latestAttempt.warnings
    : (entry?.warnings ?? []);

  /** ユニークなタグ cls を収集してバッジ用リストを返す。 */
  const allTagCls = entry
    ? [...new Set(toolCalls.flatMap((tc) => tc.tags.map((t) => t.meta.cls)))]
    : [];

  return (
    <>
      {/* 操作行: コピー / 経過時間 / ログ / 再生成（再生成のみ右端へ） */}
      <div className="flex items-center gap-0.5 -ml-1 mt-0.5 w-full">
        <CopyButton text={copyText} />
        {elapsedMs !== undefined && (
          <span
            className="text-[10px] text-ch-t4 font-mono px-1 select-none"
            title="モデルへリクエストしてから応答完了までの時間"
          >
            {formatElapsed(elapsedMs)}
          </span>
        )}
        {logMessageId && (
          <button
            onClick={handleLogToggle}
            className="flex items-center gap-1.5 text-[11px] text-ch-t4 hover:text-ch-t3 transition-colors p-1 rounded"
          >
            <span className="text-[9px] opacity-50">{logExpanded ? "▼" : "▶"}</span>
            <span>ログ</span>
            {loading && <span className="animate-pulse text-[10px]">…</span>}
            {entry?.has_error && (
              <span className="text-red-400 text-[10px] font-medium">⚠ エラー</span>
            )}
            {!loading && allTagCls.map((cls, i) => (
              <span key={i} className={`px-1 py-0.5 rounded text-[9px] ${TAG_COLORS[cls] ?? TAG_COLORS["tag-unknown"]}`}>
                {toolCalls.flatMap((tc) => tc.tags).find((t) => t.meta.cls === cls)?.meta.label ?? cls}
              </span>
            ))}
          </button>
        )}
        {onDiscard && (
          <DiscardButton onClick={onDiscard} title={discardTitle} className="ml-auto" />
        )}
        {onRegenerate && (
          <RegenerateButton
            onClick={onRegenerate}
            title={regenerateTitle}
            className={onDiscard ? "" : "ml-auto"}
          />
        )}
      </div>

      {/* ログ展開コンテンツ */}
      {logMessageId && logExpanded && (
        <div className="pt-1.5 mt-1 space-y-1.5" style={{ borderTop: "1px solid var(--ch-sep)" }}>
          {loading && (
            <div className="text-ch-t4 text-[11px] animate-pulse px-1">読み込み中...</div>
          )}
          {!loading && !entry && (
            <div className="text-ch-t4 text-[11px] px-1">ログを取得できませんでした</div>
          )}
          {entry && (
            <>
              {/* tool_calls（attempts 統合済み） */}
              {toolCalls.length > 0 && (
                <div className="space-y-1">
                  {toolCalls.map((tc, i) => (
                    <ToolCallRow
                      key={i}
                      tc={tc}
                      logMessageId={logMessageId}
                      onOpenRaw={(filename) => setRawModal(filename)}
                    />
                  ))}
                </div>
              )}

              {/* warnings（attempts 統合済み） */}
              {warnings.length > 0 && (
                <div className="space-y-0.5">
                  {warnings.map((w, i) => (
                    <div key={i} className="text-amber-400 text-[10px] flex gap-1.5 items-start">
                      <span className="shrink-0">⚠</span>
                      <span className="break-all">{w.tag}: {w.message}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* tool_callsもwarningsもない場合 */}
              {toolCalls.length === 0 && warnings.length === 0 && (
                <div className="text-ch-t4 text-[11px] px-1">ツール呼び出しなし</div>
              )}
            </>
          )}
        </div>
      )}

      {/* Raw ログモーダル */}
      {rawModal && logMessageId && (
        <RawLogModal
          messageId={logMessageId}
          filename={rawModal}
          onClose={() => setRawModal(null)}
        />
      )}
    </>
  );
}
