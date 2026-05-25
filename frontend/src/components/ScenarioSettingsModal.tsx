/**
 * シナリオセッション設定モーダル（左上のモデルチップから開く）。
 *
 * タブ切り替えで以下 2 機能を提供する:
 *   - 「モデル」タブ: シナリオ用モデル / あらすじ作成用モデルを個別に切り替える。
 *     あらすじ蒸留はレートリミット節約のために GM とは別の軽量モデルにできる。
 *   - 「あらすじ」タブ: 既存の SynopsisModal と同じ自動/手動あらすじの閲覧・編集。
 *
 * これにより、左上のモデルチップ・右上のあらすじボタンを 1 つのモーダルに統合できる。
 * （未確認の自動あらすじ更新バッジは、モデルチップ側に表示する）
 */
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { ScenarioPreset, ScenarioSynopsis } from "../api";

/** 表示中のタブ。 */
type Tab = "model" | "synopsis";

interface Props {
  /** 選択候補となる LLM プリセット一覧（fetchScenarioPresets の結果）。 */
  presets: ScenarioPreset[];
  /** 現在セッションに紐付いている GM プリセット ID。 */
  currentGmPresetId: string;
  /** 現在セッションに紐付いているあらすじ蒸留プリセット ID。 */
  currentSynopsisPresetId: string;
  /** GM プリセット変更時のコールバック。 */
  onApplyGmPreset: (presetId: string) => void;
  /** あらすじ蒸留プリセット変更時のコールバック。 */
  onApplySynopsisPreset: (presetId: string) => void;
  /** セッションのあらすじ（未取得は null）。 */
  synopsis: ScenarioSynopsis | null;
  /** あらすじを部分更新（auto / manual のどちらか / 両方）。 */
  onSynopsisChange: (patch: { auto?: string; manual?: string }) => Promise<void>;
  /** synopsis_auto への自動追記フローを手動起動する。 */
  onSynopsisRegenerate: () => Promise<void>;
  /** 送信中など、編集を無効化すべき状態。 */
  disabled: boolean;
  /** モーダルを閉じるコールバック。 */
  onClose: () => void;
  /** 初期表示するタブ（省略時は "model"）。 */
  initialTab?: Tab;
}

/** プリセット選択ボタン行。GM / Synopsis で共用する。 */
function PresetButtons({
  presets,
  selectedId,
  onSelect,
}: {
  presets: ScenarioPreset[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  if (presets.length === 0) {
    return (
      <p className="text-ch-t3 text-xs">
        LLM プリセットがありません。
        <a
          href="/ui/presets"
          target="_blank"
          rel="noopener noreferrer"
          className="underline ml-1 hover:text-ch-t2"
        >
          backend で登録
        </a>
      </p>
    );
  }
  return (
    <div className="flex gap-1.5 flex-wrap">
      {presets.map((p) => {
        const active = selectedId === p.id;
        return (
          <button
            key={p.id}
            onClick={() => onSelect(p.id)}
            className="rounded-md px-2.5 py-1 text-xs transition-colors"
            style={{
              border: `1px solid ${active ? "var(--ch-accent)" : "var(--ch-sep2)"}`,
              background: active ? "oklch(50% 0.13 226 / 0.10)" : "transparent",
              color: active ? "var(--ch-accent)" : "rgb(var(--ch-t2))",
            }}
            title={`${p.provider} / ${p.model_id || "default"}`}
          >
            {p.name}
            <span className="text-ch-t4 ml-1 font-mono text-[10px]">
              {p.provider}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/** シナリオセッション設定モーダル本体。 */
export default function ScenarioSettingsModal({
  presets,
  currentGmPresetId,
  currentSynopsisPresetId,
  onApplyGmPreset,
  onApplySynopsisPreset,
  synopsis,
  onSynopsisChange,
  onSynopsisRegenerate,
  disabled,
  onClose,
  initialTab = "model",
}: Props) {
  const [tab, setTab] = useState<Tab>(initialTab);

  /* ── モデルタブ用ローカル選択（適用ボタンで確定） ── */
  const [selGmId, setSelGmId] = useState(currentGmPresetId);
  const [selSynopsisId, setSelSynopsisId] = useState(currentSynopsisPresetId);

  /* ── あらすじタブ用ドラフト（SynopsisModal と同じ振る舞い） ── */
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

  // セッション側で外部変更があったらローカル選択も合わせる。
  useEffect(() => {
    setSelGmId(currentGmPresetId);
  }, [currentGmPresetId]);
  useEffect(() => {
    setSelSynopsisId(currentSynopsisPresetId);
  }, [currentSynopsisPresetId]);

  // Esc キーで閉じる。
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", fn);
    return () => document.removeEventListener("keydown", fn);
  }, [onClose]);

  /** 「適用」ボタン: 変更されたプリセットだけ親に通知して、モーダルを閉じる。 */
  const applyPresets = () => {
    if (selGmId && selGmId !== currentGmPresetId) {
      onApplyGmPreset(selGmId);
    }
    if (selSynopsisId && selSynopsisId !== currentSynopsisPresetId) {
      onApplySynopsisPreset(selSynopsisId);
    }
    onClose();
  };

  const presetsDirty =
    (selGmId !== "" && selGmId !== currentGmPresetId) ||
    (selSynopsisId !== "" && selSynopsisId !== currentSynopsisPresetId);

  const saveAuto = async () => {
    if (savingAuto) return;
    setSavingAuto(true);
    try {
      await onSynopsisChange({ auto: autoDraft });
    } finally {
      setSavingAuto(false);
    }
  };

  const saveManual = async () => {
    if (savingManual) return;
    setSavingManual(true);
    try {
      await onSynopsisChange({ manual: manualDraft });
    } finally {
      setSavingManual(false);
    }
  };

  const regenerate = async () => {
    if (regenerating) return;
    setRegenerating(true);
    try {
      await onSynopsisRegenerate();
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
        style={{
          border: "1px solid var(--ch-sep2)",
          boxShadow: "var(--ch-shadow)",
          maxHeight: "85vh",
        }}
      >
        {/* ヘッダー（タイトル + 閉じる） */}
        <div
          className="flex items-center px-4 py-2.5 shrink-0"
          style={{ borderBottom: "1px solid var(--ch-sep)" }}
        >
          <span className="text-ch-t1 text-xs font-semibold flex-1">
            シナリオセッション設定
          </span>
          <button
            onClick={onClose}
            className="text-ch-t3 hover:text-ch-t1 text-base px-1.5"
          >
            ✕
          </button>
        </div>

        {/* タブ */}
        <div
          className="flex shrink-0"
          style={{ borderBottom: "1px solid var(--ch-sep)" }}
        >
          {([
            { key: "model" as Tab, label: "モデル" },
            { key: "synopsis" as Tab, label: "あらすじ" },
          ]).map((t) => {
            const active = tab === t.key;
            return (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className="px-4 py-2 text-xs transition-colors"
                style={{
                  color: active ? "var(--ch-accent)" : "rgb(var(--ch-t2))",
                  borderBottom: active
                    ? "2px solid var(--ch-accent)"
                    : "2px solid transparent",
                  fontWeight: active ? 600 : 400,
                }}
              >
                {t.label}
              </button>
            );
          })}
        </div>

        {/* ボディ */}
        <div className="px-4 py-4 overflow-y-auto flex flex-col gap-4">
          {tab === "model" && (
            <>
              <div className="flex flex-col gap-1.5">
                <div className="text-[11px] text-ch-t3">
                  シナリオ用モデル（GM：会話本編を生成）
                </div>
                <PresetButtons
                  presets={presets}
                  selectedId={selGmId}
                  onSelect={setSelGmId}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <div className="text-[11px] text-ch-t3">
                  あらすじ作成用モデル（履歴を圧縮して保持。軽量モデルで節約可）
                </div>
                <PresetButtons
                  presets={presets}
                  selectedId={selSynopsisId}
                  onSelect={setSelSynopsisId}
                />
              </div>
              <div className="flex justify-end pt-2">
                <button
                  onClick={applyPresets}
                  disabled={!presetsDirty || disabled}
                  className="text-white text-xs font-medium px-3.5 py-1.5 rounded-md transition-opacity disabled:opacity-30 disabled:cursor-not-allowed"
                  style={{ background: "var(--ch-accent)" }}
                >
                  適用
                </button>
              </div>
            </>
          )}

          {tab === "synopsis" && (
            <>
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
                      disabled={
                        disabled ||
                        savingAuto ||
                        autoDraft === (synopsis?.auto ?? "")
                      }
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
                  <span>
                    補足メモ（手書き。自動更新では破壊されない・GM への補正指示）
                  </span>
                  <button
                    onClick={saveManual}
                    disabled={
                      disabled ||
                      savingManual ||
                      manualDraft === (synopsis?.manual ?? "")
                    }
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
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
