/**
 * シナリオの「これまでのあらすじ」編集モーダル。
 *
 * 記憶捏造対策として導入された 2 種類のあらすじを編集する:
 *   - synopsis_auto: LLM が古い経緯を自動で要約・追記したもの。GM プロンプトのメイン。
 *     ユーザは自由編集できる（捏造記述を発見したら削除・修正できる）。
 *   - synopsis_manual: プレイヤーが手で書き留めた補足メモ。自動更新で破壊されない。
 */
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { ScenarioSynopsis } from "../api";

interface Props {
  /** セッションのあらすじ（未取得は null）。 */
  synopsis: ScenarioSynopsis | null;
  /** あらすじを部分更新（auto / manual のどちらか / 両方）。 */
  onChange: (patch: { auto?: string; manual?: string }) => Promise<void>;
  /** synopsis_auto への自動追記フローを手動起動する。 */
  onRegenerate: () => Promise<void>;
  /** 送信中など、編集を無効化すべき状態。 */
  disabled: boolean;
  /** モーダルを閉じるコールバック。 */
  onClose: () => void;
}

/** あらすじ編集モーダル本体。 */
export default function SynopsisModal({
  synopsis,
  onChange,
  onRegenerate,
  disabled,
  onClose,
}: Props) {
  const [autoDraft, setAutoDraft] = useState(synopsis?.auto ?? "");
  const [manualDraft, setManualDraft] = useState(synopsis?.manual ?? "");
  const [savingAuto, setSavingAuto] = useState(false);
  const [savingManual, setSavingManual] = useState(false);
  const [regenerating, setRegenerating] = useState(false);

  // サーバ最新値が変わったら、未保存の編集がない側を同期する。
  useEffect(() => {
    setAutoDraft((d) => (d === "" ? synopsis?.auto ?? "" : d));
    setManualDraft((d) => (d === "" ? synopsis?.manual ?? "" : d));
  }, [synopsis]);

  // Esc キーで閉じる。
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", fn);
    return () => document.removeEventListener("keydown", fn);
  }, [onClose]);

  const saveAuto = async () => {
    if (savingAuto) return;
    setSavingAuto(true);
    try {
      await onChange({ auto: autoDraft });
    } finally {
      setSavingAuto(false);
    }
  };

  const saveManual = async () => {
    if (savingManual) return;
    setSavingManual(true);
    try {
      await onChange({ manual: manualDraft });
    } finally {
      setSavingManual(false);
    }
  };

  const regenerate = async () => {
    if (regenerating) return;
    setRegenerating(true);
    try {
      await onRegenerate();
    } finally {
      setRegenerating(false);
    }
  };

  return createPortal(
    <div
      onClick={onClose}
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: "var(--ch-overlay)" }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-[560px] bg-ch-bg rounded-xl flex flex-col overflow-hidden"
        style={{ border: "1px solid var(--ch-sep2)", boxShadow: "var(--ch-shadow)", maxHeight: "85vh" }}
      >
        {/* ヘッダー */}
        <div
          className="flex items-center px-4 py-2.5 shrink-0"
          style={{ borderBottom: "1px solid var(--ch-sep)" }}
        >
          <span className="text-ch-t1 text-xs font-semibold flex-1">これまでのあらすじ</span>
          <button onClick={onClose} className="text-ch-t3 hover:text-ch-t1 text-base px-1.5">
            ✕
          </button>
        </div>

        {/* ボディ */}
        <div className="px-4 py-4 overflow-y-auto flex flex-col gap-4">
          {/* auto: メインのあらすじ（LLM 自動生成・追記） */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-[11px] text-ch-t3">
              <span>自動あらすじ（メイン。LLM が古い履歴を要約・追記）</span>
              <div className="flex gap-2">
                <button
                  onClick={regenerate}
                  disabled={disabled || regenerating}
                  className="text-ch-t2 hover:text-ch-t1 text-[11px] px-2 py-0.5 rounded disabled:opacity-30"
                  style={{ border: "1px solid var(--ch-sep2)" }}
                  title="今すぐ古い履歴を要約して追記する"
                >
                  {regenerating ? "生成中…" : "追記更新"}
                </button>
                <button
                  onClick={saveAuto}
                  disabled={disabled || savingAuto || autoDraft === (synopsis?.auto ?? "")}
                  className="text-white text-[11px] font-medium px-2.5 py-0.5 rounded disabled:opacity-30"
                  style={{ background: "var(--ch-accent)" }}
                  title="編集内容を保存（捏造記述を削除・修正するのに使う）"
                >
                  {savingAuto ? "保存中…" : "保存"}
                </button>
              </div>
            </div>
            <textarea
              value={autoDraft}
              onChange={(e) => setAutoDraft(e.target.value)}
              rows={8}
              placeholder="（履歴が上限を超えると LLM が自動で要約・追記します）"
              className="bg-ch-s3 text-ch-t1 placeholder-ch-t3 rounded-lg px-3 py-2 text-sm resize-y focus:outline-none"
              style={{ border: "1px solid var(--ch-sep2)" }}
            />
          </div>
          {/* manual: プレイヤー手書きの補足メモ */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-[11px] text-ch-t3">
              <span>補足メモ（手書き。自動更新では破壊されない・GM への補正指示）</span>
              <button
                onClick={saveManual}
                disabled={disabled || savingManual || manualDraft === (synopsis?.manual ?? "")}
                className="text-white text-[11px] font-medium px-2.5 py-0.5 rounded disabled:opacity-30"
                style={{ background: "var(--ch-accent)" }}
              >
                {savingManual ? "保存中…" : "保存"}
              </button>
            </div>
            <textarea
              value={manualDraft}
              onChange={(e) => setManualDraft(e.target.value)}
              rows={5}
              placeholder="例: 主人公はレイカと「絶対に裏切らない」と約束した。"
              className="bg-ch-s3 text-ch-t1 placeholder-ch-t3 rounded-lg px-3 py-2 text-sm resize-y focus:outline-none"
              style={{ border: "1px solid var(--ch-sep2)" }}
            />
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
